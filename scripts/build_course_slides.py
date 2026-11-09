"""Build the editable course progress deck using standard-library OOXML.

Fallback for hosts without the presentation artifact runtime. No model, GPU,
Office installation or third-party package is required. HTML is a content/layout
preview, not proof of rendering in PowerPoint.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "汇报"
FONT = "Microsoft YaHei"
EMU = 9525
SLIDES = [
    {"title": "PageLoRA", "subtitle": "单 GPU LLM 推理优化", "cover": True,
     "lines": ["分页 KV 缓存 · 前缀复用 · CUDA Graph", "课程项目进度汇报｜2026.11.09"],
     "note": "今天汇报的是已有项目作为课程基础的适配进度。项目实现与历史实验已经存在；本轮新增的是代码梳理、基准工具修复和服务器实验方案。本课程服务器尚未产生新测量结果。"},
    {"title": "项目目标与使用场景",
     "lines": ["在一张 GPU 上运行已有大语言模型，优化请求延迟。",
               "长提示词：Prefill 计算输入，建立注意力 KV 缓存。",
               "逐 token 生成：Decode 重复读取权重与历史 KV。",
               "典型场景：共享上下文、多轮短交互、LoRA 推理。"],
     "note": "项目不是训练新模型，而是改变执行路径和缓存组织。Prefill 与 Decode 的瓶颈不同，小 batch 解码尤其需要关注启动开销和访存。来源：README.md；model/model_manager.py；model/flashinfer/generation.py。"},
    {"title": "已有的优化机制",
     "lines": ["FlashInfer：使用专门的 paged-KV attention 内核。",
               "分页 KV：管理固定大小的缓存页与请求页表。",
               "Radix 前缀缓存：检索相同 token 前缀，复用已有 KV。",
               "CUDA Graph：捕获并重放解码计算，减少提交开销。"],
     "note": "代码位置：model/flashinfer/attention_backend.py、generation.py、cuda_graph.py；model/kvcache/paged_kvcache_manager.py、flashinfer_page_table.py、radix_cache.py。缓存命中和 Graph 都不是无条件加速：页表维护、plan、捕获成本与长度会影响收益。LoRA 和显存协调保留为扩展。"},
    {"title": "课程适配：基础模型主线，LoRA 扩展",
     "lines": ["研究问题一：Graph 对单卡请求延迟有多大影响？",
               "研究问题二：共享前缀能节省多少重复计算？",
               "主实验：基础模型，同卡、同精度、同生成设置。",
               "扩展实验：LoRA 合并与 A/B 增量执行的代价。"],
     "note": "建议题目：PageLoRA：基于分页 KV 缓存与 CUDA Graph 的单 GPU LLM 推理优化。不需要重写项目。主线排除训练、多适配器调度等额外变量。HF 与 FlashInfer 的整体差异不能单独证明分页策略的收益；必须增加同后端消融或显存证据才能作该归因。"},
    {"title": "历史结果：Qwen3-8B 的 Graph 消融",
     "table": [["历史配置", "Graph off", "Graph on", "吞吐增幅"],
               ["LoRA merge", "38.72", "46.68", "+20.6%"],
               ["LoRA A/B no-merge", "29.88", "45.23", "+51.4%"]],
     "caption": "RTX 3090 24GB｜约 2K 输入 / 128 输出｜单位：decode tok/s",
     "bottom": "历史记录；尚未在课程服务器复现，不代表普遍收益。",
     "note": "来源：docs/experiment_results_infer_engine.md，Primary Results To Reuse，台账日期 2026-05-29。merge 原始值 38.7156→46.6840；no-merge 原始值 29.8762→45.2289；warmup=1、measure=3。增幅为 on/off-1。这不是本轮新测数据，也不是端到端吞吐。历史 vLLM LoRA 45.1408 tok/s，包含零适配器设置，因此不在这里主张全面超越 vLLM。"},
    {"title": "历史证据的使用边界",
     "lines": ["前缀实验曾混用冷、热缓存条件，不能直接横比。",
               "部分 LoRA 对照使用零适配器，只反映执行路径开销。",
               "初始化时间受 JIT、图捕获和磁盘缓存状态影响。",
               "原始日志主要位于被忽略目录，需补可复现证据。"],
     "note": "来源：docs/experiment_results_infer_engine.md 对 .local 日志、vLLM 零 LoRA 和热缓存的说明；test/inference/common.py 的 probe 路径。decode 计时边界也不完全一致，所以本轮主实验采用端到端延迟与吞吐。旧工程实现与本学期新增工作须按课程要求分别披露。"},
    {"title": "本轮已修复的实验工具问题",
     "lines": ["文档入口：修正 README 中无效的命令行参数。",
               "前缀计时：移除隐式单 token 探测带来的缓存预热。",
               "缓存协议：两个后端在每组序列开始前按同一策略重置。",
               "异常处理：vLLM 无法重置时显式失败，不冒充冷缓存。"],
     "note": "修改：README.md；test/inference/bench_prefix_caching.py。前缀 runner 复制 Namespace，设置 throughput_timing=total，避免 _run_vllm_once 的 decode 探测污染缓存。reset=true 对两个后端每个 warmup 和 measured sequence 生效。vLLM reset_prefix_cache 不存在或返回 False 时抛出错误。已添加回归测试，依赖 torch/pytest 的测试待服务器运行。"},
    {"title": "服务器主实验矩阵",
     "table": [["运行路径", "配置", "目的"],
               ["Hugging Face", "基础模型", "参考实现"],
               ["vLLM", "基础模型 / 默认图路径", "成熟引擎对照"],
               ["FlashInfer", "Graph off", "引擎基线"],
               ["FlashInfer", "Graph on", "单项优化消融"]],
     "caption": "输入目标 512 / 2048 / 4096 tokens；固定输出 128 tokens",
     "bottom": "每项预热 2 次、测量 5 次；串行执行，单 GPU 可见。",
     "note": "实现：scripts/run_course_experiments.py。命令见 docs/服务器实验指南.md。固定 adapter-role=base、greedy_fast、ignore_eos、compute-logprobs=false、seed=42、repetition-penalty=1.0。vLLM tensor_parallel_size=1；FlashInfer cold cache。实际输入 tokens 受模板与合成块影响，以日志为准。dtype 需要在服务器核对；启动器要求 BF16 可用以匹配 vLLM 显式设置。"},
    {"title": "前缀复用的测量协议",
     "lines": ["每组 4 个串行分支请求：共享前缀，后缀不同。",
               "冷序列：每组前清 KV，第 1 次冷，第 2–4 次复用。",
               "热序列：保留预热产生的前缀缓存，测量稳态。",
               "同后端与跨后端比较，都必须说明缓存条件。"],
     "note": "脚本 suite=prefix 覆盖两后端乘 reset true/false。这里冷是 KV 冷，不是磁盘或 JIT 冷。首请求与后续分支的差异不是严格的缓存开关因果消融，还需同 prompt 每请求清缓存的对照才能作该结论。热条件至少 warmup=1。前缀 benchmark 强制 total 路径，无单 token 探测。"},
    {"title": "指标与正确性验收",
     "lines": ["主指标：端到端请求延迟、实际输出 tokens / 总时长。",
               "辅助证据：分阶段计时、前缀命中、显存采样。",
               "正确性：HF 对照、Graph 开关、缓存复用输出一致性。",
               "稳定性：跨页长输出、部分页前缀、失败后的资源回收。"],
     "note": "当前启动器负责性能测量，不自动完成模型级正确性验收。需服务器比较首步 logits 及 token 序列，分叉时定位首 token 与数值误差。p95 在少量样本下不稳定，正式报告建议增加至 20 次并重复独立进程。nvidia-smi 是设备采样，不是严格的本进程峰值。FlashInfer trace 的 CPU 时间也不是单 kernel GPU 时间。"},
    {"title": "当前进度与待验证项",
     "lines": ["已有基础：推理引擎、缓存管理、Graph / LoRA、历史记录。",
               "本轮完成：中文说明、工具修复、实验脚本与汇报材料。",
               "本地检查：语法、两项标准库测试、命令矩阵 dry-run。",
               "待服务器验证：依赖安装、回归测试、数值正确性与性能。"],
     "note": "本地未安装 torch/pytest，也未运行模型或 GPU 测试。两项标准库测试是 test/test_course_experiments.py 的矩阵测试，不等于前缀修复已通过 GPU 回归。新增依赖型回归见 test/test_inference_bench_entrypoints.py。不要把已有项目代码算作本轮新开发内容。"},
    {"title": "下一步：先跑通一个模型",
     "lines": ["① 确认服务器 GPU、可用显存、CUDA 与模型目录。",
               "② 跑正确性测试与 512 输入 / 32 输出的冒烟实验。",
               "③ 跑主实验、Graph 消融与前缀复用矩阵。",
               "④ 根据真实日志补结果图、瓶颈解释与结论。"],
     "note": "操作入口：docs/服务器实验指南.md。最小命令：python -m scripts.run_course_experiments --model /path/to/model --gpu 0 --prompt-lengths 512 --output-tokens 32 --warmup-runs 1 --measure-runs 1。先用 --dry-run 查看命令。发回 course_results 中的 metadata.json、runs.json 和日志，再更新汇报；本页不虚构时间表和性能目标。"},
]


def xml(text):
    return escape(str(text), quote=True)


def u(value):
    return str(round(value * EMU))


def para(text, size=2200, color="173047", bold=False):
    return (f'<a:p><a:pPr/><a:r><a:rPr lang="zh-CN" sz="{size}" b="{int(bold)}">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:latin typeface="{FONT}"/><a:ea typeface="{FONT}"/>'
            f'</a:rPr><a:t>{xml(text)}</a:t></a:r>'
            f'<a:endParaRPr lang="zh-CN" sz="{size}"/></a:p>')


def textbox(idx, text, x, y, w, h, size=2200, color="173047", bold=False):
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{idx}" name="Text {idx}"/>'
            '<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr>'
            f'<a:xfrm><a:off x="{u(x)}" y="{u(y)}"/><a:ext cx="{u(w)}" cy="{u(h)}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/>'
            '<a:ln><a:noFill/></a:ln></p:spPr><p:txBody>'
            '<a:bodyPr wrap="square" lIns="0" rIns="0" tIns="0" bIns="0" anchor="t"/>'
            '<a:lstStyle/>' + ''.join(para(t, size, color, bold) for t in text.split('\n')) +
            '</p:txBody></p:sp>')


def native_table(rows, widths):
    grid = ''.join(f'<a:gridCol w="{u(w)}"/>' for w in widths)
    body = ''
    for index, row in enumerate(rows):
        cells = ''
        for content in row:
            bg, color = ("173047", "FFFFFF") if index == 0 else ("FFFFFF", "173047")
            cells += ('<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>' +
                      para(content, 2000, color, index == 0) +
                      '</a:txBody><a:tcPr marL="152400" marR="76200" marT="114300" marB="76200">'
                      f'<a:solidFill><a:srgbClr val="{bg}"/></a:solidFill></a:tcPr></a:tc>')
        body += f'<a:tr h="{u(68)}">{cells}</a:tr>'
    return ('<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="8" name="Editable evidence table"/>'
            '<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr><p:xfrm>'
            f'<a:off x="{u(80)}" y="{u(245)}"/><a:ext cx="{u(1120)}" cy="{u(68 * len(rows))}"/>'
            '</p:xfrm><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
            '<a:tbl><a:tblPr firstRow="1"/><a:tblGrid>' + grid + '</a:tblGrid>' + body +
            '</a:tbl></a:graphicData></a:graphic></p:graphicFrame>')


NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')
GROUP = ('<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
         '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
         '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>')


def slide_xml(data, page):
    cover = data.get("cover", False)
    bg, foreground = ("173047", "FFFFFF") if cover else ("F6F5F1", "173047")
    pieces = [textbox(2, data["title"], 80, 130 if cover else 65, 1120, 90,
                      5200 if cover else 3200, foreground, True)]
    if cover:
        pieces += [textbox(3, data["subtitle"], 80, 250, 1120, 80, 3600, "73D5C3")]
        for i, line in enumerate(data["lines"]):
            pieces.append(textbox(4 + i, line, 80, 390 + i * 70, 1120, 60, 2200, "FFFFFF"))
    elif "table" in data:
        pieces.append(textbox(3, data["caption"], 80, 163, 1120, 55, 1800, "496172"))
        widths = [420, 240, 240, 220] if len(data["table"][0]) == 4 else [300, 450, 370]
        pieces.append(native_table(data["table"], widths))
        pieces.append(textbox(9, data["bottom"], 80, 620, 1120, 42, 1700, "496172"))
    else:
        for i, line in enumerate(data["lines"]):
            pieces.append(textbox(3 + i, line, 80, 210 + i * 97, 1120, 77, 2200, foreground))
    pieces.append(textbox(20, f"{page:02d} / {len(SLIDES):02d}", 1100, 678, 105, 30, 1200,
                          "A6BEC9" if cover else "607581"))
    return (f'<p:sld {NS}><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="{bg}"/>'
            '</a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree>' + GROUP + ''.join(pieces) +
            '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


def rels(entries):
    prefix = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
    return ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
            ''.join(f'<Relationship Id="{key}" Type="{prefix}{kind}" Target="{target}"/>'
                    for key, kind, target in entries) + '</Relationships>')


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    files["_rels/.rels"] = rels([("rId1", "officeDocument", "ppt/presentation.xml")])
    slide_ids = ''.join(f'<p:sldId id="{256+i}" r:id="rId{i+2}"/>' for i in range(len(SLIDES)))
    files["ppt/presentation.xml"] = (f'<p:presentation {NS}><p:sldMasterIdLst>'
        '<p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>' + slide_ids +
        f'</p:sldIdLst><p:sldSz cx="{u(1280)}" cy="{u(720)}" type="screen16x9"/>'
        '<p:notesSz cx="6858000" cy="9144000"/><p:defaultTextStyle/></p:presentation>')
    files["ppt/_rels/presentation.xml.rels"] = rels(
        [("rId1", "slideMaster", "slideMasters/slideMaster1.xml")] +
        [(f"rId{i+2}", "slide", f"slides/slide{i+1}.xml") for i in range(len(SLIDES))])
    cmap = ('<p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" '
            'accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" '
            'hlink="hlink" tx1="dk1" tx2="dk2"/>')
    files["ppt/slideMasters/slideMaster1.xml"] = (
        f'<p:sldMaster {NS}><p:cSld><p:spTree>{GROUP}</p:spTree></p:cSld>{cmap}'
        '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
        '<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>')
    files["ppt/slideMasters/_rels/slideMaster1.xml.rels"] = rels(
        [("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml"), ("rId2", "theme", "../theme/theme1.xml")])
    files["ppt/slideLayouts/slideLayout1.xml"] = (
        f'<p:sldLayout {NS} type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>'
        f'{GROUP}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
    files["ppt/slideLayouts/_rels/slideLayout1.xml.rels"] = rels(
        [("rId1", "slideMaster", "../slideMasters/slideMaster1.xml")])
    colors = dict(dk1="173047", lt1="FFFFFF", dk2="496172", lt2="F6F5F1", accent1="117C77",
                  accent2="73D5C3", accent3="A47A2C", accent4="607581", accent5="3457A0",
                  accent6="AD5F67", hlink="117C77", folHlink="607581")
    color_xml = ''.join(f'<a:{key}><a:srgbClr val="{value}"/></a:{key}>' for key, value in colors.items())
    font = f'<a:latin typeface="{FONT}"/><a:ea typeface="{FONT}"/><a:cs typeface="Arial"/>'
    files["ppt/theme/theme1.xml"] = (
        f'<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="PageLoRA">'
        f'<a:themeElements><a:clrScheme name="PageLoRA">{color_xml}</a:clrScheme>'
        f'<a:fontScheme name="Chinese"><a:majorFont>{font}</a:majorFont><a:minorFont>{font}</a:minorFont></a:fontScheme>'
        '<a:fmtScheme name="Simple"><a:fillStyleLst>' +
        '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>' * 3 +
        '</a:fillStyleLst><a:lnStyleLst>' +
        '<a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>' * 3 +
        '</a:lnStyleLst><a:effectStyleLst>' + '<a:effectStyle><a:effectLst/></a:effectStyle>' * 3 +
        '</a:effectStyleLst><a:bgFillStyleLst>' +
        '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>' * 3 +
        '</a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>')
    for i, data in enumerate(SLIDES, start=1):
        files[f"ppt/slides/slide{i}.xml"] = slide_xml(data, i)
        files[f"ppt/slides/_rels/slide{i}.xml.rels"] = rels([
            ("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rId2", "notesSlide", f"../notesSlides/notesSlide{i}.xml")])
        files[f"ppt/notesSlides/notesSlide{i}.xml"] = (
            f'<p:notes {NS}><p:cSld><p:spTree>{GROUP}' +
            '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Speaker notes"/><p:cNvSpPr/>'
            '<p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/>'
            '<p:txBody><a:bodyPr/><a:lstStyle/>' + para(data["note"], 1200) +
            '</p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>')
        files[f"ppt/notesSlides/_rels/notesSlide{i}.xml.rels"] = rels(
            [("rId1", "slide", f"../slides/slide{i}.xml")])
    overrides = {"ppt/presentation.xml": "presentation.main", "ppt/slideMasters/slideMaster1.xml": "slideMaster",
                 "ppt/slideLayouts/slideLayout1.xml": "slideLayout", "ppt/theme/theme1.xml": "theme"}
    overrides.update({f"ppt/slides/slide{i}.xml": "slide" for i in range(1, len(SLIDES)+1)})
    overrides.update({f"ppt/notesSlides/notesSlide{i}.xml": "notesSlide" for i in range(1, len(SLIDES)+1)})
    files["[Content_Types].xml"] = (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>' + ''.join(
            f'<Override PartName="/{part}" ContentType="application/vnd.openxmlformats-officedocument.'
            f'{"theme" if kind == "theme" else "presentationml." + kind}+xml"/>'
            for part, kind in overrides.items()) + '</Types>')
    target = OUT / "PageLoRA_课程进度汇报.pptx"
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for name, content in files.items():
            ET.fromstring(content)  # Reject malformed XML before writing.
            archive.writestr(name, '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + content)
    preview = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PageLoRA 进度汇报预览</title><style>',
               'body{margin:0;background:#d9dee1;font-family:"Microsoft YaHei",sans-serif}',
               'section{box-sizing:border-box;position:relative;width:1280px;height:720px;margin:24px auto;background:#F6F5F1;color:#173047;padding:65px 80px;overflow:hidden}',
               'h1{font-size:42.667px;line-height:1.4;margin:0;height:90px;font-weight:700}',
               '.lines{position:absolute;top:210px;left:80px;width:1120px}.lines p{font-size:29.333px;line-height:1.5;height:77px;margin:0 0 20px}',
               '.cover{background:#173047;color:white;padding-top:130px}.cover h1{font-size:69.333px}.cover h2{font-size:48px;color:#73D5C3;margin:30px 0 55px;font-weight:400}',
               '.cover p{font-size:29.333px;margin:28px 0}.caption{font-size:24px;margin:8px 0 34px;color:#496172}',
               'table{border-collapse:collapse;width:1120px;table-layout:fixed;font-size:26.667px;background:white}td,th{text-align:left;box-sizing:border-box;height:68px;padding:12px 16px}th{background:#173047;color:white}',
               '.bottom{position:absolute;top:620px;left:80px;font-size:22.667px;color:#496172}.page{position:absolute;left:1100px;top:678px;font-size:16px;color:#607581}',
               '@media print{body{background:white}section{margin:0;page-break-after:always}}', '</style>']
    notes = ['# PageLoRA 课程进度汇报讲稿', '', 'PPT 内含相同讲稿与来源说明；本稿不包含课程服务器新测数据。', '']
    for i, data in enumerate(SLIDES, start=1):
        preview += [f'<section id="slide-{i}" class="{"cover" if data.get("cover") else ""}"><h1>{xml(data["title"])}</h1>']
        if data.get("cover"):
            preview += [f'<h2>{xml(data["subtitle"])}</h2>'] + [f'<p>{xml(t)}</p>' for t in data["lines"]]
        elif "table" in data:
            preview += [f'<p class="caption">{xml(data["caption"])}</p><table>']
            widths = [420, 240, 240, 220] if len(data["table"][0]) == 4 else [300, 450, 370]
            preview += ['<colgroup>' + ''.join(f'<col style="width:{w}px">' for w in widths) + '</colgroup>']
            for j, row in enumerate(data["table"]):
                tag = 'th' if j == 0 else 'td'
                preview += ['<tr>' + ''.join(f'<{tag}>{xml(t)}</{tag}>' for t in row) + '</tr>']
            preview += [f'</table><p class="bottom">{xml(data["bottom"])}</p>']
        else:
            preview += ['<div class="lines">'] + [f'<p>{xml(t)}</p>' for t in data['lines']] + ['</div>']
        preview += [f'<span class="page">{i:02d} / {len(SLIDES):02d}</span></section>']
        notes += [f'## {i}. {data["title"]}', '', data['note'], '']
    (OUT / "预览.html").write_text('\n'.join(preview) + '</html>', encoding='utf-8')
    (OUT / "讲稿.md").write_text('\n'.join(notes), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    build()
