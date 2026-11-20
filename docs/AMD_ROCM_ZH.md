# AMD ROCm 适配

适用入口：`indextts.infer_v2`、`indextts.infer_v2_5` 和 `webui.py`。
模型权重不需要转换；ROCm PyTorch 同样使用 `cuda` 设备名称。

## 安装

使用单独的 Python 3.11 环境和项目要求的 PyTorch 2.8。
以下安装方式忽略项目中的 NVIDIA 下载源，选择 ROCm 6.4 软件包：

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --no-sources --torch-backend rocm6.4 -e '.[webui]'
```

此环境使用 `uv pip` 管理。不要直接执行普通 `uv sync` 或 `uv run`，
它们会按根目录的 CUDA 锁文件重新同步环境。运行时直接使用 `.venv/bin/python`。
系统 ROCm 版本与 PyTorch wheel 的 HIP 运行库版本可以不同，是否兼容需要实测。

## 不加载权重的 GPU 检查

```bash
.venv/bin/python tools/rocm_check.py
.venv/bin/python tools/rocm_check.py --dtype float16
.venv/bin/python tools/rocm_check.py --dtype bfloat16
```

检查包含矩阵乘法、卷积、注意力和 FFT，并与 CPU 计算结果比较。
需能访问 `/dev/kfd` 和 `/dev/dri/renderD*`；受限沙箱可能隐藏这些设备。
检查通过并不代表完整语音推理已经通过。

## 下载 IndexTTS-2.5 并启动

在仓库根目录执行：

```bash
uv tool run --from modelscope modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir checkpoints
.venv/bin/python webui.py --version 2.5 --model_dir checkpoints
```

首次启动还会下载辅助模型。先使用默认精度验证短文本；BF16 运算检查通过后，
可尝试 WebUI 的 `--fp16` 参数（2.5 入口实际启用 BF16）。
低显存设备默认跳过 Qwen 情感文本模型，仍可使用情感参考音频或向量。

ROCm 下自动关闭 BigVGAN 自定义 CUDA 内核，使用现有 PyTorch 实现。
目前拒绝启用 `use_deepspeed` 和 `use_accel`，以避免进入本分支尚未验证的加速路径。
初次验证不要启用 `--torch_compile`。TensorRT 后端不属于此适配范围。

## 本机基线

2026-11-20：Fedora 44，Ryzen 7 8845H / Radeon 780M (`gfx1103`)，
系统内存约 27 GiB，GPU 预留内存 4 GiB，系统 ROCm/HIP 7.1.1。
GTT 是共享内存，不代表同等容量的独立显存。
系统 HIP 7.1.1 已通过设备识别、4 KiB GPU 内存分配、写入、同步和读回检查。
主模型文件已下载到 `checkpoints/`，版本为 2.5。

已安装 `torch==2.8.0+rocm6.4` 和 `torchaudio==2.8.0+rocm6.4`。
该 wheel 未提供 `gfx1103` 的 rocBLAS/Tensile 内核：不设置覆盖参数时，
第一次矩阵乘法会因缺少内核而终止。

在本机临时使用 `HSA_OVERRIDE_GFX_VERSION=11.0.0` 后，FP32 和 BF16
矩阵乘法、卷积、注意力及 FP32 FFT 检查通过，与 CPU 参考值在脚本容差内一致：

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/python tools/rocm_check.py
HSA_OVERRIDE_GFX_VERSION=11.0.0 .venv/bin/python tools/rocm_check.py --dtype bfloat16
```

这是让软件选择 `gfx1100` 内核的本机兼容性实验，不会改变真实硬件架构，
也不是所有算子均受支持的保证。不在 Python 源码或全局环境中自动设置。
在 `gfx1100` 覆盖下，FP16 检查在卷积阶段以
`HSA_STATUS_ERROR_INVALID_ISA` 终止。这不是硬件不支持 FP16 的结论。

随后改用用户发现的 `gfx1102` 覆盖，顺序运行 FP32、FP16、BF16 检查，
三种精度均通过矩阵乘法、卷积、注意力及 FP32 FFT 的 CPU 对比检查。
后续本机完整推理优先测试此设置：

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.2 .venv/bin/python tools/rocm_check.py --dtype float32
HSA_OVERRIDE_GFX_VERSION=11.0.2 .venv/bin/python tools/rocm_check.py --dtype float16
HSA_OVERRIDE_GFX_VERSION=11.0.2 .venv/bin/python tools/rocm_check.py --dtype bfloat16
```

2.5 的 `--fp16` 参数实际选择 BF16，需注意二者区别。
完整项目依赖已安装，`uv pip check` 通过；推理模块、NumPy、Librosa、Gradio
导入通过。安装完整依赖后的 BF16 基础运算复测也通过。

完整合成检查使用项目的 `tests/sample_prompt.wav` 作为参考音频，关闭 Qwen
情感文本模型和可选加速。辅助模型准备好后可运行：

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.2 .venv/bin/python tools/rocm_synthesize.py
```

输出为 `outputs/rocm_smoke.wav`，同目录 JSON 记录耗时、音频时长、RTF 和
PyTorch GPU 内存峰值。文件非空和非静音检查不代替人工试听。

首次完整 BF16 合成已通过，文本为“你好，这是在AMD显卡上进行的语音合成测试。”。
输出为 22,050 Hz、约 4.55 秒的非静音 WAV；加载耗时约 40.87 秒，合成耗时
150.98 秒，RTF 约 33.17，PyTorch 分配内存峰值约 5.33 GiB，预留峰值约
5.47 GiB。这是单次首次运行结果，不代表稳定性能，也尚未人工确认音质。
其中 GPT 约 5.57 秒、S2Mel 约 13.90 秒、BigVGAN 约 123.22 秒，
日志含 MIOpen workspace 警告，声码器是当前主要耗时阶段。

另一个新进程重复相同文本也成功：模型加载约 13.61 秒，合成约 90.84 秒，
音频仍为约 4.55 秒，RTF 约 19.96，内存峰值不变。BigVGAN 仍耗时约
70.97 秒，因此不能把首次运行的全部耗时都归因于缓存或编译。当前仅验证
了此短中文样例的两次 BF16 合成；FP32 完整推理、长文本及 WebUI 交互尚未验证。
重复输出为 `outputs/rocm_smoke_repeat.wav`。

本机 WebUI 使用同样的兼容参数和 BF16：

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.2 .venv/bin/python webui.py --version 2.5 --model_dir checkpoints --fp16 --host 127.0.0.1
```
