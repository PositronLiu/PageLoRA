"""GPU probe shared by CLI runners, with no torch dependency in the parent."""
import json
import os
import subprocess
import sys
from pathlib import Path


def configure_single_gpu_environment(env):
    """Configure local-only workers without modifying the host networking."""
    env.setdefault('VLLM_HOST_IP', '127.0.0.1')
    if Path('/sys/class/net/lo').exists():
        env.setdefault('GLOO_SOCKET_IFNAME', 'lo')
    if Path('/usr/lib/x86_64-linux-gnu/libcuda.so.1').exists():
        env.setdefault('TRITON_LIBCUDA_PATH', '/usr/lib/x86_64-linux-gnu')
    return env


def choose_dtype(requested, bf16_supported):
    if requested == 'auto':
        return 'bfloat16' if bf16_supported else 'float16'
    if requested == 'bfloat16' and not bf16_supported:
        raise ValueError('Selected GPU does not support BF16; use --dtype float16')
    return requested


def probe_gpu(gpu, requested='auto'):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    code = (
        "import json,torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "p=torch.cuda.get_device_properties(0); "
        "print(json.dumps(dict(name=p.name, capability=list(torch.cuda.get_device_capability(0)), "
        "memory_gib=p.total_memory/2**30, bf16=p.major>=8 and torch.cuda.is_bf16_supported(), "
        "torch=torch.__version__, cuda=torch.version.cuda)))"
    )
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('GPU probe failed:\n' + result.stderr)
    info = json.loads(result.stdout.strip().splitlines()[-1])
    info['dtype'] = choose_dtype(requested, info['bf16'])
    return info
