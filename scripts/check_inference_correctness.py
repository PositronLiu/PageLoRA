"""Isolated base-model greedy comparisons on identical token IDs (requires server GPU)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('hf', 'flashinfer_eager', 'flashinfer_graph', 'vllm')


def compare_sequences(reference, candidate):
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return {'equal': False, 'first_difference': index, 'reference_token': left, 'candidate_token': right}
    equal = len(reference) == len(candidate)
    return {'equal': equal, 'first_difference': None if equal else min(len(reference), len(candidate)),
            'reference_length': len(reference), 'candidate_length': len(candidate)}


def compare_logits(reference, candidate, atol, rtol):
    if not reference or len(reference) != len(candidate):
        return {'available': False, 'reason': 'missing logits or vocabulary size differs', 'allclose': False}
    if not all(math.isfinite(v) for v in reference + candidate):
        return {'available': True, 'allclose': False, 'reason': 'non-finite logits'}
    errors = [abs(a-b) for a,b in zip(reference, candidate)]
    top = sorted(range(len(reference)), key=reference.__getitem__, reverse=True)[:2]
    return {'available': True, 'allclose': all(e <= atol + rtol * abs(a) for a,e in zip(reference, errors)),
            'max_abs_error': max(errors), 'rmse': math.sqrt(sum(e*e for e in errors)/len(errors)),
            'argmax_equal': max(range(len(candidate)), key=candidate.__getitem__) == top[0],
            'reference_top1_top2_margin': reference[top[0]]-reference[top[1]] if len(top)>1 else None,
            'atol': atol, 'rtol': rtol}


def worker(args):
    import torch
    from test.inference.common import build_parser, _make_flashinfer_manager, _make_hf_manager, _make_vllm_llm
    torch.manual_seed(42)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This correctness matrix requires CUDA and BF16')
    cases = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    config = build_parser().parse_args(['--model',args.model, '--adapter-role','base', '--lora-dropout','0',
        '--sampling-mode','greedy_fast', '--compute-logprobs','false', '--repetition-penalty','1',
        '--ignore-eos','--max-new-tokens',str(args.output_tokens)])
    results=[]
    if args.worker == 'vllm':
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        config.vllm_dtype='bfloat16'
        config.vllm_max_model_len=args.max_model_len
        config.vllm_gpu_memory_utilization=.8
        llm=_make_vllm_llm(config)
        for case in cases:
            outputs=llm.generate([TokensPrompt(prompt_token_ids=case['ids'])],
                sampling_params=SamplingParams(temperature=0, max_tokens=args.output_tokens,
                                               ignore_eos=True, repetition_penalty=1), use_tqdm=False)
            results.append({'case':case['name'], 'tokens':list(outputs[0].outputs[0].token_ids),
                            'first_logits':None, 'logits_note':'vLLM public generation API: sequence-only comparison'})
    else:
        from model.generation import generate_with_logprobs
        is_hf=args.worker=='hf'
        config.flashinfer_cuda_graph_decode=args.worker=='flashinfer_graph'
        manager=(_make_hf_manager if is_hf else _make_flashinfer_manager)(config,torch.device('cuda:0'))
        capture=[]
        # Explicit teacher-forced first step. Do not instrument CUDA Graph with hooks.
        if is_hf:
            for case in cases:
                with manager.use_adapter(role='base'), manager.inference_session(), torch.no_grad():
                    outputs=manager.model(input_ids=torch.tensor([case['ids']],device='cuda'),use_cache=False)
                    capture.append(outputs.logits[0,-1].float().cpu().tolist())
                    del outputs
        else:
            # Capture the first prefill logits through the generation helper,
            # before any graph capture; immediately remove the temporary hook.
            import model.flashinfer.generation as generation
            original=generation._model_final_norm_and_head
            from model.flashinfer.cuda_graph import FlashInferDecodeCudaGraphRunner
            original_run=FlashInferDecodeCudaGraphRunner.run
            replay_count=[0]
            def record_replay(self, **kwargs):
                output=original_run(self, **kwargs)
                if output is not None:
                    replay_count[0]+=1
                return output
            FlashInferDecodeCudaGraphRunner.run=record_replay
            def collect(model, hidden, **kwargs):
                logits=original(model,hidden,**kwargs)
                if not capture:
                    capture.append(logits[0,-1].detach().float().cpu().tolist())
                return logits
        for index,case in enumerate(cases):
            if not is_hf:
                replay_start=replay_count[0]
                manager.kvcache_manager.invalidate_adapter('general')
                capture.clear()
                generation._model_final_norm_and_head=collect
            try:
                cold=generate_with_logprobs(manager,case['ids'],adapter_name='general',adapter_role='base',
                    max_new_tokens=args.output_tokens,temperature=0,ignore_eos=True,
                    sampling_mode='greedy_fast',compute_logprobs=False,repetition_penalty=1)
            finally:
                if not is_hf:
                    generation._model_final_norm_and_head=original
            row={'case':case['name'], 'tokens':cold.response_ids,
                 'first_logits':capture[index] if is_hf else (capture[0] if capture else None)}
            if not is_hf:
                hot=generate_with_logprobs(manager,case['ids'],adapter_name='general',adapter_role='base',
                    max_new_tokens=args.output_tokens,temperature=0,ignore_eos=True,
                    sampling_mode='greedy_fast',compute_logprobs=False,repetition_penalty=1)
                row['reuse_tokens']=hot.response_ids
                row['reuse_match_tokens']=(hot.timings or {}).get('cache_match_tokens')
                runner=getattr(manager,'_flashinfer_decode_graph_runner',None)
                row['graph_requested']=config.flashinfer_cuda_graph_decode
                row['graph_successful_replays']=replay_count[0]-replay_start
                row['graph_disabled_reason']=getattr(runner,'disabled_reason',None)
            results.append(row)
    import importlib.metadata
    packages={}
    for package in ['torch','transformers','flashinfer-python','vllm']:
        try:
            packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package]=None
    Path(args.result).write_text(json.dumps({'variant':args.worker,'model':args.model,
        'dtype':'bfloat16','gpu':torch.cuda.get_device_name(0), 'packages':packages,
        'results':results}),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--output-tokens',type=int,default=32)
    parser.add_argument('--max-model-len',type=int,default=4096)
    parser.add_argument('--output-dir',default='course_results')
    parser.add_argument('--atol',type=float,default=.1)
    parser.add_argument('--rtol',type=float,default=.01)
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    parser.add_argument('--worker',choices=VARIANTS,help=argparse.SUPPRESS)
    parser.add_argument('--cases',help=argparse.SUPPRESS)
    parser.add_argument('--result',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.output_tokens<1 or args.atol<0 or args.rtol<0 or not all(map(math.isfinite,[args.atol,args.rtol])):
        parser.error('Positive output length and finite non-negative tolerances required')
    if args.worker:
        worker(args)
        return
    if ',' in args.gpu or not args.gpu.strip():
        parser.error('Select exactly one physical GPU')
    if 'hf' not in args.variants:
        parser.error('HF reference must be included')
    if len(set(args.variants))<2:
        parser.error('Select HF and at least one candidate backend')
    from transformers import AutoTokenizer
    from data.tokenizer_utils import encode_chat
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    _,ids=encode_chat(tokenizer,[{'role':'user','content':'Explain why a KV cache accelerates autoregressive decoding.'}],enable_thinking=False)
    # Exact token boundaries exercise full and partial pages without downloading datasets.
    cases=[{'name':f'prefix_{length}', 'ids':(ids*((length+len(ids)-1)//len(ids)))[:length]}
           for length in (31,32,33,129)]
    if max(len(c['ids']) for c in cases)+args.output_tokens>args.max_model_len:
        parser.error('--max-model-len is too small')
    destination=Path(args.output_dir).resolve()/('correctness_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    destination.mkdir(parents=True)
    cases_path=destination/'cases.json'
    cases_path.write_text(json.dumps(cases),encoding='utf-8')
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,MINI_OPENCLAW_FLASHINFER_PERF_LOG='1',
               MINI_OPENCLAW_FLASHINFER_TORCH_COMPILE='0', MINI_OPENCLAW_FLASHINFER_COMPILE_MLP='0',
               MINI_OPENCLAW_FLASHINFER_COMPILE_POST='0',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    (destination/'metadata.json').write_text(json.dumps({'args':vars(args),'python':sys.version,
        'commit':revision.stdout.strip(),'git_error':revision.stderr.strip(),
        'environment':{k:v for k,v in env.items() if k.startswith(('CUDA_','MINI_OPENCLAW_','VLLM_'))}},indent=2),encoding='utf-8')
    outcomes=[]; reports={}
    for variant in dict.fromkeys(args.variants):
        result=destination/f'{variant}.json'
        command=[sys.executable,'-m','scripts.check_inference_correctness','--model',args.model,
                 '--worker',variant,'--cases',str(cases_path),'--result',str(result),
                 '--output-tokens',str(args.output_tokens),'--max-model-len',str(args.max_model_len)]
        print('Checking',variant,flush=True)
        with (destination/f'{variant}.log').open('w',encoding='utf-8') as stream:
            completed=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        outcomes.append({'variant':variant,'returncode':completed.returncode,'command':command})
        if completed.returncode==0:
            reports[variant]=json.loads(result.read_text(encoding='utf-8'))['results']
        (destination/'runs.json').write_text(json.dumps(outcomes,indent=2),encoding='utf-8')
    checks=[]
    reference={r['case']:r for r in reports.get('hf',[])}
    for variant,rows in reports.items():
        if variant=='hf': continue
        for row in rows:
            ref=reference.get(row['case'])
            if ref is None: continue
            check={'variant':variant,'case':row['case'],
                   'sequence':compare_sequences(ref['tokens'],row['tokens']),
                   'fixed_output_length':len(row['tokens'])==len(ref['tokens'])==args.output_tokens}
            if row.get('first_logits') is not None:
                check['first_logits']=compare_logits(ref['first_logits'],row['first_logits'],args.atol,args.rtol)
            if 'reuse_tokens' in row:
                check['cache_reuse']=compare_sequences(row['tokens'],row['reuse_tokens'])
                check['reuse_match_tokens']=row.get('reuse_match_tokens')
                check['reuse_exercised']=(row.get('reuse_match_tokens') or 0)>0
            if variant=='flashinfer_graph':
                check['graph_successful_replays']=row.get('graph_successful_replays',0)
                check['graph_exercised']=check['graph_successful_replays']>0
                check['graph_disabled_reason']=row.get('graph_disabled_reason')
            checks.append(check)
    eager={r['case']:r for r in reports.get('flashinfer_eager',[])}
    for row in reports.get('flashinfer_graph',[]):
        if row['case'] in eager:
            ref=eager[row['case']]
            checks.append({'variant':'graph_vs_eager','case':row['case'],
                           'sequence':compare_sequences(ref['tokens'],row['tokens']),
                           'fixed_output_length':len(ref['tokens'])==len(row['tokens'])==args.output_tokens})
    passed=bool(checks) and all(o['returncode']==0 for o in outcomes) and all(
        c['sequence']['equal'] and c['fixed_output_length'] and c.get('cache_reuse',{'equal':True})['equal']
        and c.get('first_logits',{'allclose':True})['allclose'] and c.get('reuse_exercised',True)
        and c.get('graph_exercised',True) for c in checks)
    report={'strict_pass':passed,'checks':checks,'runs':outcomes,
            'limitations':['vLLM compares greedy token IDs only, not full logits.',
                'First-step logits cover prefill; later steps compare sequences, not teacher-forced logits.',
                'A mismatch requires inspecting first divergence and floating-point margins; it alone does not prove a bug.',
                'These cases cover repeated prompts and page boundaries, not concurrent serving or OOM recovery.',
                'Graph validation requires a nonzero count of successful runner replays.',
                'BF16 tolerances default to atol=0.1, rtol=0.01; these are screening settings, not universal error bounds.']}
    (destination/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Correctness report','',f'Strict pass: {passed}', '',
           '| Candidate (reference: HF unless noted) | Case | Exact tokens | First divergence | Logits allclose | Cache reused | Graph replayed |',
           '|---|---|---|---|---|---|---|']
    for c in checks:
        lines.append(f"| {c['variant']} | {c['case']} | {c['sequence']['equal']} | "
                     f"{c['sequence'].get('first_difference')} | {c.get('first_logits',{}).get('allclose','n/a')} | "
                     f"{c.get('reuse_exercised','n/a')} | {c.get('graph_exercised','n/a')} |")
    lines+=['','## Scope','']+[f'- {limitation}' for limitation in report['limitations']]
    (destination/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Strict pass:',passed,'Report:',destination/'report.json')
    if not passed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
