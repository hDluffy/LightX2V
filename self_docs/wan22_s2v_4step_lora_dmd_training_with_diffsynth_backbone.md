# Wan2.2-S2V 4-Step LoRA 蒸馏训练流程

日期：2026-08-10

## 1. 结论

本仓库已补齐 Wan2.2-S2V 4-step LoRA DMD 蒸馏训练的默认 backbone 接入：训练配置现在直接使用 `lightx2v_train.model_zoo.wan_s2v:DiffSynthWanS2VBackbone`，不再要求用户额外实现 `your_package.wan_s2v_train:WanS2VBackbone`。

实现方式是复用 DiffSynth-Studio 的 trainable Wan2.2-S2V DiT 和 `model_fn_wans2v`，LightX2V 负责 S2V 条件缓存、DMD 训练编排、LoRA 注入/保存，以及训练后用 LightX2V S2V 4-step 推理验证。

参考来源：

- DiffSynth-Studio Wan2.2-S2V LoRA 示例：<https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/model_training/lora/Wan2.2-S2V-14B.sh>
- DiffSynth WanVideo pipeline / S2V model function：<https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/pipelines/wan_video.py>
- DiffSynth 模型训练文档：<https://diffsynth-studio-doc.readthedocs.io/en/latest/Pipeline_Usage/Model_Training.html>
- DiffSynth ModelConfig / loader 文档：<https://diffsynth-studio-doc.readthedocs.io/en/latest/API_Reference/core/loader.html>
- DiffSynth Wan 模型说明：<https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Wan.html>

## 2. 代码补充内容

### 2.1 DiffSynth-backed S2V backbone

文件：`lightx2v_train/lightx2v_train/model_zoo/wan_s2v.py`

新增类：`DiffSynthWanS2VBackbone`

能力：

- 从 DiffSynth-Studio 导入 `WanVideoPipeline`、`ModelConfig`、`model_fn_wans2v`。
- 自动从 `pretrained_model_name_or_path` 下查找 `diffusion_pytorch_model*.safetensors`。
- 支持显式 `model_paths`，可以是单文件、glob 字符串、shard 列表或 DiffSynth `ModelConfig` kwargs。
- 支持 `model_id_with_origin_paths`，例如 `Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors`。
- 将 LightX2V cache 格式转换成 DiffSynth S2V forward 格式。
- 暴露 `blocks` 属性，方便 FSDP2 按 DiT block shard。
- 默认要求 per-rank `batch_size=1`，避免 S2V 音频/pose 条件 batch 语义不一致。

格式桥接：

```text
LightX2V DMD noisy latent: [B, C, F, H, W]
LightX2V ref_latents:      [B, C, 1, H, W]
DiffSynth S2V input:       cat([ref_latents, noisy_latent], dim=2)
DiffSynth output:          [B, C, 1+F, H, W]
adapter return:            output[:, :, 1:] -> [B, C, F, H, W]
```

### 2.2 Unconditional audio 置零

文件：`lightx2v_train/lightx2v_train/model_zoo/wan_s2v.py`

`WanS2VTrainModel.encode_condition(..., unconditional=True)` 会把 `audio_input` 置零。这样 teacher CFG 的 unconditional 分支与 S2V 推理语义一致：负向分支保留图像、motion、pose 条件，但不使用音频驱动。

### 2.3 默认训练 YAML 已切到 DiffSynth backbone

文件：`lightx2v_train/configs/train/dmd/wan22_s2v_4step_lora.yaml`

关键配置：

```yaml
model:
  name: wan_s2v
  pretrained_model_name_or_path: /path/to/Wan2.2-S2V-14B
  transformer_class_path: lightx2v_train.model_zoo.wan_s2v:DiffSynthWanS2VBackbone
  transformer_factory: from_pretrained
  transformer_init_kwargs:
    model_paths: null
    model_id_with_origin_paths: null
    load_device: cpu
    use_gradient_checkpointing: true
    use_gradient_checkpointing_offload: false
    strict_batch_size_one: true
```

如果 `/path/to/Wan2.2-S2V-14B` 目录下能找到 `diffusion_pytorch_model*.safetensors`，`model_paths` 可以保持 `null`。

### 2.4 训练脚本支持 DiffSynth 源码路径

文件：`lightx2v_train/scripts/run_wan22_s2v_4step_lora.sh`

新增环境变量：

```bash
DIFFSYNTH_STUDIO_PATH=/path/to/DiffSynth-Studio
```

设置后脚本会把该路径加入 `PYTHONPATH`。如果已通过 pip 安装 DiffSynth-Studio，可以不设置。

## 3. 训练原理

### 3.1 为什么需要 DiffSynth backbone

LightX2V 的 `lightx2v.models.networks.wan.s2v_model.WanS2VModel` 是推理优化栈，权重放在自定义 weight module 中，核心路径带 `torch.no_grad()`，适合高性能推理，不适合 PEFT LoRA 注入和 DMD 反传。

DiffSynth-Studio 提供 trainable `torch.nn.Module` 形式的 WanVideo DiT。这里不复制 DiffSynth 的模型源码，而是在训练时导入 DiffSynth 的 S2V backbone 和 S2V model function，使 LightX2V 训练框架能够对其注入 LoRA 并反传。

### 3.2 4-step DMD LoRA 蒸馏

训练使用已有 `S2VDmdLoraTrainer`，包含三个角色：

- student：待训练的 S2V LoRA，按固定 4-step schedule 生成短步轨迹。
- fake：辅助 LoRA 模型，用于估计 student 当前生成分布。
- teacher：冻结的 Wan2.2-S2V 原模型，用 CFG 提供高质量参考 score。

固定 4-step：

```text
[1000, 750, 500, 250]
```

训练目标是让 student 在这 4 个 timestep 上学到接近 teacher 多步/CFG 轨迹的 velocity，因此推理时可以关闭 CFG，用 4 次 denoiser 调用完成生成。

### 3.3 为什么先做 S2V cache

S2V 条件包括文本、参考图、音频、motion、可选 pose。训练前先用 LightX2V S2V runner 构造 `.pt` 缓存，可以减少重复编码，并保证训练条件预处理和推理路径一致。

## 4. 环境准备

### 4.1 LightX2V 训练环境

```bash
cd /data/hjq/LightX2V

conda create -n lightx2v-train python=3.11 -y
conda activate lightx2v-train

# 按本机 CUDA 版本选择 torch。示例为 CUDA 12.8。
conda install -c nvidia -c pytorch \
  pytorch torchvision torchaudio pytorch-cuda=12.8 -y

python -m pip install -U pip packaging ninja cmake
python -m pip install -r requirements.txt
python -m pip install peft omegaconf PyYAML safetensors decord soundfile librosa
python -m pip install -v -e .
```

### 4.2 安装或挂载 DiffSynth-Studio

方式 A：源码路径加入 `PYTHONPATH`。

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git /path/to/DiffSynth-Studio
export DIFFSYNTH_STUDIO_PATH=/path/to/DiffSynth-Studio
export PYTHONPATH=${DIFFSYNTH_STUDIO_PATH}:${PYTHONPATH}
```

方式 B：pip 安装。

```bash
python -m pip install git+https://github.com/modelscope/DiffSynth-Studio.git
```

import 检查：

```bash
python - <<'PY'
from diffsynth import ModelConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline, model_fn_wans2v
print('DiffSynth import ok')
PY
```

### 4.3 常用环境变量

```bash
cd /data/hjq/LightX2V
export PYTHONPATH=/data/hjq/LightX2V/lightx2v_train:/data/hjq/LightX2V:${PYTHONPATH}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

## 5. 权重准备

需要两类权重：

- LightX2V 推理缓存构造用的 Wan2.2-S2V-14B 原模型目录。
- DiffSynth trainable DiT 加载用的 Wan2.2-S2V diffusion safetensors。

自动检测方式：

```yaml
model:
  pretrained_model_name_or_path: /path/to/Wan2.2-S2V-14B
  transformer_init_kwargs:
    model_paths: null
```

显式 glob：

```yaml
model:
  pretrained_model_name_or_path: /path/to/Wan2.2-S2V-14B
  transformer_init_kwargs:
    model_paths: /path/to/Wan2.2-S2V-14B/diffusion_pytorch_model*.safetensors
```

显式列表：

```yaml
model:
  transformer_init_kwargs:
    model_paths:
      - /path/to/diffusion_pytorch_model-00001-of-00006.safetensors
      - /path/to/diffusion_pytorch_model-00002-of-00006.safetensors
```

DiffSynth remote pattern：

```yaml
model:
  transformer_init_kwargs:
    model_paths: null
    model_id_with_origin_paths:
      - Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors
```

## 6. 数据准备与 cache 构造

### 6.1 metadata

JSONL 示例：

```jsonl
{"prompt":"a person is speaking to camera", "image_path":"/data/s2v/images/000001.jpg", "audio_path":"/data/s2v/audio/000001.wav", "negative_prompt":" ", "src_pose_path":""}
{"prompt":"a person is singing", "image_path":"/data/s2v/images/000002.jpg", "audio_path":"/data/s2v/audio/000002.wav", "negative_prompt":" ", "src_pose_path":"/data/s2v/pose/000002.mp4"}
```

字段：

- `prompt`：正向文本。
- `image_path`：参考图。
- `audio_path`：驱动音频。
- `negative_prompt`：负向文本，可用空格。
- `src_pose_path`：可选 pose 视频；没有 pose 时写空字符串或不写。

建议拆成：

```text
/path/to/wan22_s2v_train.jsonl
/path/to/wan22_s2v_val.jsonl
```

### 6.2 构造 train cache

```bash
cd /data/hjq/LightX2V/lightx2v_train

MODEL_PATH=/path/to/Wan2.2-S2V-14B \
CONFIG_JSON=../configs/wan22/wan_s2v.json \
METADATA_PATH=/path/to/wan22_s2v_train.jsonl \
OUTPUT_DIR=./data_cache/wan22_s2v/train \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/build_wan22_s2v_cache.sh --overwrite
```

### 6.3 构造 val cache

```bash
MODEL_PATH=/path/to/Wan2.2-S2V-14B \
CONFIG_JSON=../configs/wan22/wan_s2v.json \
METADATA_PATH=/path/to/wan22_s2v_val.jsonl \
OUTPUT_DIR=./data_cache/wan22_s2v/val \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/build_wan22_s2v_cache.sh --overwrite
```

调试时限制样本：

```bash
bash scripts/build_wan22_s2v_cache.sh --max_samples 2 --overwrite
```

缓存 `.pt` 应包含：

```text
context
context_null
ref_latents
motion_latents
cond_latents
audio_input
motion_frames
latent_shape
drop_motion_frames
add_last_motion
```

## 7. 修改训练配置

编辑：`lightx2v_train/configs/train/dmd/wan22_s2v_4step_lora.yaml`

至少修改这些路径：

```yaml
model:
  pretrained_model_name_or_path: /path/to/Wan2.2-S2V-14B
  fake:
    pretrained_model_name_or_path: /path/to/Wan2.2-S2V-14B
  teacher:
    pretrained_model_name_or_path: /path/to/Wan2.2-S2V-14B

data:
  train:
    cache_path:
      - /data/hjq/LightX2V/lightx2v_train/data_cache/wan22_s2v/train
  val:
    cache_path:
      - /data/hjq/LightX2V/lightx2v_train/data_cache/wan22_s2v/val
```

核心训练参数：

```yaml
training:
  method: s2v_dmd_lora
  train_type: lora
  max_train_iters: 1000
  gradient_accumulation_iters: 1
  dmd:
    num_inference_steps: 4
    denoising_step_list: [1000, 750, 500, 250]
    fake_update_ratio: 5
    timestep_shift: 5.0
  lora:
    rank: 64
    alpha: 64
    target_modules: [q, k, v, o, ffn.0, ffn.2]
```

DiffSynth 官方 Wan2.2-S2V LoRA 示例也使用同类 attention/FFN target module。这里保持 `q,k,v,o,ffn.0,ffn.2`，便于训练产物和 LightX2V 推理 LoRA loader 对齐。

## 8. Smoke test

先临时改小配置：

```yaml
training:
  max_train_iters: 1
  save_every_iters: 1

data:
  train:
    max_samples: 1
  val:
    max_samples: 1
```

单卡只建议做接口验证，14B DMD 三模型很可能显存不足：

```bash
cd /data/hjq/LightX2V/lightx2v_train

CONFIG=configs/train/dmd/wan22_s2v_4step_lora.yaml \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
DIFFSYNTH_STUDIO_PATH=/path/to/DiffSynth-Studio \
bash scripts/run_wan22_s2v_4step_lora.sh
```

推荐多卡 FSDP2：

```bash
CONFIG=configs/train/dmd/wan22_s2v_4step_lora.yaml \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
DIFFSYNTH_STUDIO_PATH=/path/to/DiffSynth-Studio \
bash scripts/run_wan22_s2v_4step_lora.sh
```

通过标志：

- 能 import DiffSynth。
- 日志能加载 `DiffSynthWanS2VBackbone`。
- LoRA 注入后 trainable params 非 0。
- 能跑完 `iter=1/1`。
- 生成 `checkpoint-000000001/pytorch_lora_weights.safetensors`。

## 9. 正式训练

恢复配置：

```yaml
training:
  max_train_iters: 1000
  save_every_iters: 100
  save_total_limit: 10

data:
  train:
    max_samples: null
```

启动：

```bash
cd /data/hjq/LightX2V/lightx2v_train

CONFIG=configs/train/dmd/wan22_s2v_4step_lora.yaml \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
DIFFSYNTH_STUDIO_PATH=/path/to/DiffSynth-Studio \
bash scripts/run_wan22_s2v_4step_lora.sh
```

断点续训：

```yaml
resume:
  auto_resume: true
```

训练输出：

```text
lightx2v_train/output_train/wan22_s2v_4step_lora/
  checkpoint-000000100/
    pytorch_lora_weights.safetensors
    fake_lora/pytorch_lora_weights.safetensors
    config.yaml
    training_state.pt 或 dist_state/
```

推理验证只使用 student LoRA：

```text
checkpoint-xxxxxx/pytorch_lora_weights.safetensors
```

`fake_lora` 只用于 DMD 训练/恢复，不用于最终部署。

## 10. 4-step 推理验证

编辑：`configs/wan22/wan_s2v_distill_lora_4step.json`

```json
{
  "scheduler_type": "WanS2VStepDistillScheduler",
  "infer_steps": 4,
  "sample_shift": 5.0,
  "sample_guide_scale": 1.0,
  "enable_cfg": false,
  "denoising_step_list": [1000, 750, 500, 250],
  "lora_configs": [
    {
      "path": "/data/hjq/LightX2V/lightx2v_train/output_train/wan22_s2v_4step_lora/checkpoint-000001000/pytorch_lora_weights.safetensors",
      "strength": 1.0
    }
  ]
}
```

运行：

```bash
cd /data/hjq/LightX2V

python -m lightx2v.infer \
  --model_cls wan2.2_s2v \
  --task s2v \
  --model_path /path/to/Wan2.2-S2V-14B \
  --config_json /data/hjq/LightX2V/configs/wan22/wan_s2v_distill_lora_4step.json \
  --prompt "A person speaks naturally to the camera with subtle head motion." \
  --image_path /path/to/ref_image.jpg \
  --audio_path /path/to/audio.wav \
  --save_result_path /data/hjq/LightX2V/save_results/output_lightx2v_wan22_s2v_4step_lora.mp4
```

带 pose：

```bash
  --src_pose_path /path/to/pose.mp4
```

检查日志：

- `Using WanS2VStepDistillScheduler`
- step 日志为 `1 / 4` 到 `4 / 4`
- LoRA 成功应用
- 输出 mp4 写入并 mux 音频成功

## 11. 常见问题

### 11.1 ImportError: DiffSynthWanS2VBackbone requires DiffSynth-Studio

没有安装 DiffSynth，或源码路径没加入 `PYTHONPATH`。

处理：

```bash
export DIFFSYNTH_STUDIO_PATH=/path/to/DiffSynth-Studio
export PYTHONPATH=${DIFFSYNTH_STUDIO_PATH}:${PYTHONPATH}
```

### 11.2 DiffSynth WanVideoPipeline did not load a DiT

`pretrained_model_name_or_path` 下没有 DiT safetensors，或 `model_paths` 配错。

处理：

```yaml
model:
  transformer_init_kwargs:
    model_paths: /path/to/Wan2.2-S2V-14B/diffusion_pytorch_model*.safetensors
```

### 11.3 batch_size 报错

默认 `strict_batch_size_one=true`。保持：

```yaml
data:
  train:
    batch_size: 1
```

多卡通过 `NPROC_PER_NODE` 扩大总 batch。

### 11.4 显存不足

Wan2.2-S2V-14B DMD 会加载 student、fake、teacher 三个 DiT 角色，即便 LoRA 训练也需要大量显存。

建议：

- 使用 8 卡 FSDP2。
- 保持 `batch_size=1`。
- 保持 `gradient_checkpointing=true`。
- 先用 `max_train_iters=1` 验证。
- 降低分辨率或 `max_area` 时，需要重新构造 cache 并保持训练/推理一致。

### 11.5 推理 LoRA 无效果或加载 warning

可能是 LoRA key 与 LightX2V 推理权重 key 不匹配。

检查：

- 训练 target modules 是否仍是 `q,k,v,o,ffn.0,ffn.2`。
- LoRA 权重是否是 student checkpoint 下的 `pytorch_lora_weights.safetensors`，不是 `fake_lora`。
- 推理配置 `lora_configs[0].path` 是否为绝对路径。

## 12. 最小执行清单

```bash
# 1. 环境
cd /data/hjq/LightX2V
conda activate lightx2v-train
export DIFFSYNTH_STUDIO_PATH=/path/to/DiffSynth-Studio
export PYTHONPATH=${DIFFSYNTH_STUDIO_PATH}:/data/hjq/LightX2V/lightx2v_train:/data/hjq/LightX2V:${PYTHONPATH}

# 2. train cache
cd /data/hjq/LightX2V/lightx2v_train
MODEL_PATH=/path/to/Wan2.2-S2V-14B \
CONFIG_JSON=../configs/wan22/wan_s2v.json \
METADATA_PATH=/path/to/wan22_s2v_train.jsonl \
OUTPUT_DIR=./data_cache/wan22_s2v/train \
bash scripts/build_wan22_s2v_cache.sh --overwrite

# 3. val cache
MODEL_PATH=/path/to/Wan2.2-S2V-14B \
CONFIG_JSON=../configs/wan22/wan_s2v.json \
METADATA_PATH=/path/to/wan22_s2v_val.jsonl \
OUTPUT_DIR=./data_cache/wan22_s2v/val \
bash scripts/build_wan22_s2v_cache.sh --overwrite

# 4. 修改 configs/train/dmd/wan22_s2v_4step_lora.yaml 的模型路径和 cache 路径

# 5. 训练
CONFIG=configs/train/dmd/wan22_s2v_4step_lora.yaml \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
bash scripts/run_wan22_s2v_4step_lora.sh

# 6. 将 student LoRA 填入 configs/wan22/wan_s2v_distill_lora_4step.json

# 7. 推理验证
cd /data/hjq/LightX2V
python -m lightx2v.infer   --model_cls wan2.2_s2v   --task s2v   --model_path /path/to/Wan2.2-S2V-14B   --config_json /data/hjq/LightX2V/configs/wan22/wan_s2v_distill_lora_4step.json   --prompt "A person speaks naturally to the camera."   --image_path /path/to/ref.jpg   --audio_path /path/to/audio.wav   --save_result_path /data/hjq/LightX2V/save_results/wan22_s2v_4step_lora.mp4
```
