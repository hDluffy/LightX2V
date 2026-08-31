# LightX2V 训练环境、Wan2.2 4-step LoRA 蒸馏与 S2V 支持分析

生成时间：2026-07-06

## 1. 工程结构结论

这个仓库可以分成两套相对独立的系统：

- `lightx2v/`：推理与部署主框架。`python -m lightx2v.infer`、`LightX2VPipeline`、Wan2.2 MoE、Wan2.2-S2V、LoRA 加载、蒸馏 4 步推理都在这里。
- `lightx2v_train/`：训练框架。入口是 `lightx2v_train/train.py`，使用 YAML 配置，支持 `flow`、`dmd`、`video_dmd`、`video_ar_dmd`、`teacher_forcing`、`dopsd` 等 trainer。
- `configs/`、`scripts/`：推理配置和推理脚本为主，包含 `configs/distill/wan22/*` 与 `scripts/wan22/distill/*`。
- `lightx2v_train/configs/train/`：训练配置。当前只有 Wan2.1 T2V 相关训练配置，没有 Wan2.2 / Wan2.2-S2V 的训练配置。

关键判断：

- 推理侧已经支持 Wan2.2 MoE 4-step distilled LoRA：`model_cls=wan2.2_moe_distill`，配置在 `configs/distill/wan22/`。
- 训练侧没有开箱即用的 Wan2.2 4-step LoRA 蒸馏训练闭环。
- 训练侧 `VideoDmdTrainer` 明确限制 `model.name == "wan_t2v"` 且 `training.train_type == "full"`，因此当前不能直接训练 Wan2.2 4-step LoRA。
- Wan2.2-S2V 推理存在：`model_cls=wan2.2_s2v`，但没有 `wan2.2_s2v_distill`、没有 S2V 4-step scheduler 配置、没有 S2V LoRA 蒸馏训练数据集/模型适配。

## 2. Conda 训练环境安装流程

下面流程按仓库当前默认 Docker 的 PyTorch 2.8 / CUDA 12.8 路线整理。若使用 5090/Blackwell 或 CUDA 13，需要参考 `dockerfiles/Dockerfile_cu130` 的 torch 2.11 + CUDA 13 路线，并重新匹配 flash-attn / SageAttention wheel。

### 2.1 基础环境

```bash
conda create -n lightx2v-train python=3.11 -y
conda activate lightx2v-train

conda install -c nvidia -c pytorch \
  pytorch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 pytorch-cuda=12.8 -y

conda install -c conda-forge ffmpeg=8.0.0 git-lfs -y
git lfs install
```

### 2.2 Python 包

```bash
cd /data/hjq/LightX2V

python -m pip install -U pip
python -m pip install -U packaging ninja cmake scikit-build-core uv meson
python -m pip install -r requirements.txt

# 训练侧直接 import，但不完整出现在 requirements.txt 中的依赖
python -m pip install peft omegaconf PyYAML huggingface_hub sentencepiece pandas soundfile librosa

# 可选：使用 causal_forcing_lmdb_dataset 时需要
python -m pip install lmdb

# 安装 lightx2v 推理包；lightx2v_train 训练脚本通过 PYTHONPATH 使用
python -m pip install -v -e .
```

训练时建议设置：

```bash
cd /data/hjq/LightX2V
export PYTHONPATH=/data/hjq/LightX2V/lightx2v_train:/data/hjq/LightX2V:${PYTHONPATH}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

### 2.3 可选算子

训练 Wan native 模型可以回退到 PyTorch SDPA，但性能会明显低于 flash-attn。推理配置常用 `flash_attn3`、`sage_attn2` 等算子，建议按 GPU 架构选择安装：

- A100/4090/H100 常规路线：参考 `dockerfiles/Dockerfile`，安装 flash-attn、SageAttention、MagiAttention、q8_kernels。
- 5090/Blackwell/CUDA 13 路线：参考 `dockerfiles/Dockerfile_cu130`，使用 torch 2.11、CUDA 13、flash-attn 对应 wheel、SageAttention-1104 等。
- 只做训练功能验证时，可以先不装所有推理加速算子；若配置里写了 `sage_attn2` / `flash_attn3`，需要改成当前环境可用的 attention 类型或补装算子。

### 2.4 训练命令模板

Wan2.1 T2V DMD 示例：

```bash
cd /data/hjq/LightX2V/lightx2v_train

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export CONFIG=configs/train/dmd/wan2_1_t2v_1_3b_dmd.yaml

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train.py --config "${CONFIG}"
```

单卡验证：

```bash
cd /data/hjq/LightX2V/lightx2v_train
export CUDA_VISIBLE_DEVICES=0
torchrun --standalone --nproc_per_node=1 train.py --config configs/train/flow/wan2_1_t2v_1_3b_lora.yaml
```

## 3. 当前训练框架的数据构造

### 3.1 Wan T2V 视频监督数据

对应数据集：`wan_t2v_video_dataset`。

元数据可以是 CSV、JSON、JSONL。默认列：

- `video`：视频相对路径或绝对路径。
- `caption`：prompt。

CSV 示例：

```csv
video,caption
000001.mp4,A young woman walks through a rainy city street at night.
000002.mp4,A drone shot of snowy mountains under sunrise.
```

配置关键字段：

```yaml
data:
  train:
    name: wan_t2v_video_dataset
    data_path:
      - /path/to/train.csv
    video_root: /path/to/videos
    video_column: video
    prompt_column: caption
    height: 480
    width: 832
    num_frames: 81
    frame_rate: 24
    fix_frame_rate: false
    random_start: false
    prompt_dropout_rate: 0.1
```

数据加载逻辑会中心裁剪并 resize 到目标分辨率，输出 `video` 与 `prompt`。

### 3.2 Wan T2V 缓存数据

对应数据集：`wan_t2v_cached_dataset`。

每个 `.pt` 至少包含：

```python
{
    "latent": torch.Tensor,        # [C, F, H, W] 或 [1, C, F, H, W]
    "prompt_embed": torch.Tensor,  # [L, D] 或 [1, L, D]
    "prompt": "...",
    "video_path": "..."
}
```

配置：

```yaml
data:
  train:
    name: wan_t2v_cached_dataset
    cache_path:
      - /path/to/cache_dir
```

### 3.3 DMD / step distill prompt 数据

Wan2.1 `video_dmd` 示例使用 `prompt_dataset`，不是视频监督数据。每条数据只需要 prompt，因为 DMD 是用教师模型得分函数约束学生分布。

txt 示例：

```txt
A cinematic shot of a golden retriever running across a beach.
A macro video of rain drops sliding down a green leaf.
```

JSONL 示例：

```jsonl
{"prompt": "A cinematic shot of a golden retriever running across a beach.", "height": 480, "width": 832}
{"prompt": "A macro video of rain drops sliding down a green leaf.", "height": 720, "width": 1280}
```

## 4. Wan2.2 4-step LoRA 蒸馏：当前可执行的是推理部署

### 4.1 I2V 单卡 LoRA 4 步推理

脚本：`scripts/wan22/distill/run_wan22_moe_i2v_distill_lora_4step.sh`

配置：`configs/distill/wan22/wan_moe_i2v_distill_with_lora.json`

关键配置：

```json
{
  "infer_steps": 4,
  "target_video_length": 81,
  "target_height": 720,
  "target_width": 1280,
  "sample_guide_scale": [3.5, 3.5],
  "sample_shift": 5.0,
  "enable_cfg": false,
  "boundary_step_index": 2,
  "denoising_step_list": [1000, 750, 500, 250],
  "lora_configs": [
    {
      "name": "high_noise_model",
      "path": "/path/to/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors",
      "strength": 1.0
    },
    {
      "name": "low_noise_model",
      "path": "/path/to/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
      "strength": 1.0
    }
  ]
}
```

注意：仓库示例里的 `lightx2v/Wan2.2-Distill-Loras/...` 应替换为本机 `.safetensors` 路径；LoRA 加载代码最终使用 `safe_open` 读取本地文件。

执行：

```bash
lightx2v_path=/data/hjq/LightX2V
model_path=/path/to/Wan2.2-I2V-A14B

export CUDA_VISIBLE_DEVICES=0
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
  --model_cls wan2.2_moe_distill \
  --task i2v \
  --model_path "${model_path}" \
  --config_json ${lightx2v_path}/configs/distill/wan22/wan_moe_i2v_distill_with_lora.json \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，低质量，畸形，杂乱背景" \
  --image_path ${lightx2v_path}/assets/inputs/imgs/img_0.jpg \
  --save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_moe_i2v_distill_lora.mp4
```

### 4.2 T2V 单卡 LoRA 4 步推理

脚本：`scripts/wan22/distill/run_wan22_moe_t2v_distill_lora_4step.sh`

配置：`configs/distill/wan22/wan_moe_t2v_distill_lora.json`

关键差异：

- `target_height=480`
- `target_width=832`
- `sample_guide_scale=[4.0, 3.0]`
- high/low LoRA 分别为 T2V LoRA。

执行：

```bash
lightx2v_path=/data/hjq/LightX2V
model_path=/path/to/Wan2.2-T2V-A14B

export CUDA_VISIBLE_DEVICES=0
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
  --model_cls wan2.2_moe_distill \
  --task t2v \
  --model_path "${model_path}" \
  --config_json ${lightx2v_path}/configs/distill/wan22/wan_moe_t2v_distill_lora.json \
  --prompt "Two anthropomorphic cats in comfy boxing gear fight on a spotlighted stage." \
  --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，低质量，畸形，杂乱背景" \
  --save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_moe_t2v_distill_lora.mp4
```

### 4.3 多卡 Ulysses/CFG 并行推理

I2V：

```bash
lightx2v_path=/data/hjq/LightX2V
model_path=/path/to/Wan2.2-I2V-A14B

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source ${lightx2v_path}/scripts/base/base.sh

torchrun --nproc_per_node=8 -m lightx2v.infer \
  --model_cls wan2.2_moe_distill \
  --task i2v \
  --model_path "${model_path}" \
  --config_json ${lightx2v_path}/configs/distill/wan22/wan_moe_i2v_distill_lora_4step_cfg_ulysses.json \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，低质量，畸形，杂乱背景" \
  --image_path ${lightx2v_path}/assets/inputs/imgs/img_0.jpg \
  --save_result_path ${lightx2v_path}/save_results/wan22_i2v_lora_4step_ulysses.mp4
```

配置里的并行字段：

```json
{
  "parallel": {
    "seq_p_size": 4,
    "seq_p_attn_type": "ulysses",
    "cfg_p_size": 2
  }
}
```

实际 world size 必须等于 `seq_p_size * cfg_p_size`。

## 5. 如果目标是“训练 Wan2.2 4-step LoRA 蒸馏”，当前缺口

当前仓库没有直接可运行的 Wan2.2 4-step LoRA 蒸馏训练配置和模型适配。具体缺口如下：

1. `lightx2v_train/model_zoo/__init__.py` 只注册了 `wan_t2v` / `wan_t2v_ar`，没有 Wan2.2 MoE、Wan2.2 I2V、Wan2.2-S2V 的训练模型。
2. `VideoDmdTrainer` 只允许 `allowed_model_names = {"wan_t2v"}`。
3. `VideoDmdTrainer.__init__` 强制 `training.train_type == "full"`，不能直接走 LoRA。
4. 训练数据集只有 Wan T2V 视频/latent/prompt 数据；没有 I2V 的首帧/图像条件数据集，也没有 S2V 的音频/参考图/pose 条件数据集。
5. 推理侧的 Wan2.2 MoE high/low 分支逻辑在 `lightx2v/models/runners/wan/wan_distill_runner.py`，但训练侧没有同构的 high/low 双模型训练封装。

因此：当前能部署现成 Wan2.2 distilled LoRA；不能只改 YAML 就训练出 Wan2.2 4-step distilled LoRA。

## 6. Wan2.2 4-step LoRA 蒸馏训练参数建议

如果补齐训练侧支持，建议从仓库已有参数继承：

### 6.1 MoE 分支

Wan2.2 MoE 需要分别训练/保存两个 LoRA：

- `high_noise_model`：前半程高噪声分支。
- `low_noise_model`：后半程低噪声分支。
- `boundary_step_index=2`：4 步中前 2 步用 high，后 2 步用 low。

推理固定匹配：

```yaml
num_inference_steps: 4
denoising_step_list: [1000, 750, 500, 250]
sample_shift: 5.0
enable_cfg: false
boundary_step_index: 2
```

### 6.2 LoRA

建议起步：

```yaml
lora:
  rank: 64
  alpha: 64
  target_modules:
    - q
    - k
    - v
    - o
    - ffn.0
    - ffn.2
```

推理侧 LoRA 权重名会映射到 `self_attn.q/k/v/o`、`ffn.0/2` 等线性层。若训练侧使用 native Wan 模块，PEFT target module 名需要与 native module 的实际 `named_modules()` 对齐。

### 6.3 DMD / fake-score 参数

从现有 Wan2.1 DMD 与 Qwen/Flux LoRA DMD 配置综合，起步参数：

```yaml
training:
  method: video_moe_dmd_lora   # 需要新增
  train_type: lora
  max_train_iters: 1000
  gradient_accumulation_iters: 1
  gradient_checkpointing: true
  max_grad_norm: 1.0
  lr_scheduler: constant
  lr_warmup_iters: 10
  save_every_iters: 100
  save_total_limit: 10

  dmd:
    num_inference_steps: 4
    fake_update_ratio: 2        # LoRA DMD 先用 2；full Wan2.1 示例使用 5
    denoising_step_list: [1000, 750, 500, 250]
    warp_denoising_step: true
    num_train_timestep: 1000
    timestep_shift: 5.0
    min_step_ratio: 0.02
    max_step_ratio: 0.98
    renoise_sigma_min: 0.02
    renoise_sigma_max: 1.0
    renoise_discrete_samples: 1000
    renoise_shift: 5.0

  student:
    optimizer:
      learning_rate: 0.0001
      adam_beta1: 0.9
      adam_beta2: 0.999
      weight_decay: 0.001
      adam_epsilon: 0.00000001

  fake:
    optimizer:
      learning_rate: 0.00002
      adam_beta1: 0.9
      adam_beta2: 0.999
      weight_decay: 0.001
      adam_epsilon: 0.00000001

  teacher:
    guidance_scale: 3.5   # I2V 可从 3.5 起步；T2V 可按 high/low [4.0, 3.0] 设计
    negative_prompt: " "
    cfg_norm: none
```

全量 Wan2.1 DMD 用 `student lr=2e-6`、`fake lr=4e-7`。LoRA DMD 通常可用更高学习率，仓库里的 Qwen/Flux LoRA DMD 用 `student lr=1e-4`、`fake lr=2e-5`。

## 7. Wan2.2-S2V 4-step LoRA 蒸馏支持判断

结论：当前不支持开箱即用的 Wan2.2-S2V 4-step LoRA 蒸馏训练，也没有现成 S2V 4-step LoRA 推理配置。

已有能力：

- `scripts/wan22/run_wan22_s2v.sh` 可以跑 40 步 S2V。
- `configs/wan22/wan_s2v.json` 默认是：
  - `infer_steps=40`
  - `sample_shift=3`
  - `sample_guide_scale=4.5`
  - `enable_cfg=true`
- `WanS2VRunner` 使用 `WanS2VScheduler`，底层是 UniPC 多步调度。
- `WanS2VModel` 继承通用 `WanModel`，部分 LoRA 动态注册机制可以复用。

缺失能力：

- 没有 `wan2.2_s2v_distill` runner。
- 没有 S2V 版 4-step scheduler 配置。
- `WanS2VRunner.load_transformer()` 没有读取 `lora_configs` 并调用 `build_wan_model_with_lora`。
- S2V 推理模型是轻量 weight-module 推理栈，很多路径带 `torch.no_grad()`，不等价于训练可反传模型。
- `lightx2v_train` 没有 S2V trainable model、S2V dataset、音频条件编码缓存、参考图/pose 条件处理。

## 8. 支持 Wan2.2-S2V 4-step LoRA 蒸馏的方案

### 8.1 推理侧最小支持

目标：先支持加载已经训练好的 S2V 4-step LoRA 并用 4 步推理。

需要修改：

1. 新增 scheduler：
   - 文件：`lightx2v/models/schedulers/wan/s2v/step_distill_scheduler.py`
   - 逻辑可参考 `WanStepDistillScheduler`，但保留 S2V 输入/输出形状和 `WanS2VScheduler.prepare_clip()` 的 latent 初始化。
   - 使用 `denoising_step_list=[1000,750,500,250]` 与训练时 `sample_shift`。

2. 修改 runner：
   - 文件：`lightx2v/models/runners/wan/wan_s2v_runner.py`
   - 支持 `scheduler_type == "WanS2VStepDistillScheduler"`。
   - 在 `load_transformer()` 中读取 `lora_configs`。
   - 如果 `lora_dynamic_apply=true`，给 `WanS2VModel` 传入 `lora_path/lora_strength`。
   - 如果非动态合并，使用 `LoraAdapter` 合并到 `original_weight_dict` 后再 `_apply_weights()`。

3. 新增配置：

```json
{
  "infer_steps": 4,
  "num_train_timesteps": 1000,
  "target_fps": 16,
  "target_video_length": 81,
  "infer_frames": 80,
  "motion_frames": 73,
  "drop_first_motion": true,
  "max_area": 720896,
  "text_len": 512,
  "sample_shift": 5.0,
  "sample_guide_scale": 1.0,
  "enable_cfg": false,
  "scheduler_type": "WanS2VStepDistillScheduler",
  "denoising_step_list": [1000, 750, 500, 250],
  "lora_configs": [
    {
      "path": "/path/to/wan22_s2v_4step_lora_rank64.safetensors",
      "strength": 1.0
    }
  ]
}
```

如果 S2V 后续也拆成 high/low 噪声双分支，则按 Wan2.2 MoE 的 `name=high_noise_model/low_noise_model` 机制扩展；当前仓库里的 S2V 是单模型路径，不是 high/low MoE 目录结构。

### 8.2 训练侧完整支持

目标：训练 S2V 4-step LoRA。

需要新增或改造：

1. `lightx2v_train/model_zoo/wan_s2v.py`
   - 提供可训练 `nn.Module` 形式的 S2V transformer。
   - 封装 VAE、T5、Wav2Vec/audio encoder、参考图编码、pose 编码。
   - 实现训练框架要求的接口：
     - `encode_to_latent(sample)`
     - `encode_condition(sample)`
     - `prepare_denoiser_input(noisy_latent, condition)`
     - `denoise(denoiser_input, timestep_or_sigma, condition)`
     - `postprocess_denoiser_output(...)`
     - `prepare_infer_latents(...)`
     - `decode_latent(...)`

2. `lightx2v_train/data/s2v_dataset.py`
   - JSONL/CSV 字段建议：

```jsonl
{"video": "videos/000001.mp4", "image": "images/000001.jpg", "audio": "audio/000001.wav", "prompt": "A person is speaking to camera.", "pose": "pose/000001.mp4"}
```

   - 输出 raw 模式：
     - `video`
     - `ref_image`
     - `audio_path` 或 `audio`
     - `prompt`
     - `pose_video` 可选
   - 输出 cached 模式：
     - `latent`
     - `prompt_embed`
     - `context_null`
     - `audio_emb`
     - `ref_latents`
     - `cond_latents`
     - `motion_latents`
     - `pose_latents` 可选

3. `S2VDmdLoraTrainer`
   - 可继承 `DmdTrainer` 或抽出 `VideoDmdTrainer` 的视频 step-distill 逻辑。
   - 移除 `VideoDmdTrainer` 的 `train_type == "full"` 限制，或者新建只面向 LoRA 的 trainer。
   - `student`、`fake` 都注入 LoRA；`teacher` 冻结全量权重。
   - CFG 蒸馏时，teacher conditional branch 使用音频/文本/图像条件，unconditional branch 应与 S2V 推理逻辑一致：文本走 negative prompt，音频可置零。

4. 训练配置示例：

```yaml
model:
  name: wan_s2v
  pretrained_model_name_or_path: /path/to/Wan2.2-S2V-14B
  running_dtype: bf16
  transformer_param_dtype: bf16
  vae_dtype: fp32
  load_vae: true
  load_text_encoder: true
  load_audio_encoder: true
  load_transformer: true
  fake:
    name: wan_s2v
    pretrained_model_name_or_path: /path/to/Wan2.2-S2V-14B
  teacher:
    name: wan_s2v
    pretrained_model_name_or_path: /path/to/Wan2.2-S2V-14B

data:
  train:
    name: wan_s2v_cached_dataset
    data_path:
      - /path/to/s2v_cache
    batch_size: 1
    num_workers: 4
    shuffle: true

scheduler:
  num_train_timesteps: 1000
  timestep_distribution: uniform
  min_t: 0.001
  max_t: 1.0
  time_shift_settings:
    do_time_shift: true
    shift_type: linear
    time_shift_mu: 5.0
    time_shift_power: 1.0
    sigma_min: 0.0
    extra_one_step: true

training:
  method: s2v_dmd_lora
  train_type: lora
  max_train_iters: 1000
  gradient_accumulation_iters: 1
  gradient_checkpointing: true
  max_grad_norm: 1.0
  lr_scheduler: constant
  lr_warmup_iters: 10
  save_every_iters: 100
  save_total_limit: 10
  dmd:
    num_inference_steps: 4
    denoising_step_list: [1000, 750, 500, 250]
    warp_denoising_step: true
    fake_update_ratio: 2
    num_train_timestep: 1000
    timestep_shift: 5.0
    min_step_ratio: 0.02
    max_step_ratio: 0.98
    renoise_sigma_min: 0.02
    renoise_sigma_max: 1.0
    renoise_discrete_samples: 1000
    renoise_shift: 5.0
  lora:
    rank: 64
    alpha: 64
    target_modules:
      - q
      - k
      - v
      - o
      - ffn.0
      - ffn.2
      # 可选：音频注入层也训练，需先验证稳定性
      # - audio_injector.injector.*.q
      # - audio_injector.injector.*.k
      # - audio_injector.injector.*.v
      # - audio_injector.injector.*.o
  student:
    optimizer:
      learning_rate: 0.0001
      adam_beta1: 0.9
      adam_beta2: 0.999
      weight_decay: 0.001
      adam_epsilon: 0.00000001
  fake:
    optimizer:
      learning_rate: 0.00002
      adam_beta1: 0.9
      adam_beta2: 0.999
      weight_decay: 0.001
      adam_epsilon: 0.00000001
  teacher:
    guidance_scale: 4.5
    negative_prompt: "画面模糊，最差质量，细节模糊不清，字幕，畸形，杂乱背景"
    cfg_norm: none
  output_dir: ./output_train/wan22_s2v_4step_lora

inference:
  method: wan_s2v_infer
  num_inference_steps: 4
  denoising_step_list: [1000, 750, 500, 250]
  enable_cfg: false
  cfg_guidance_scale: 1.0
  output_dir: ./output_infer/wan22_s2v_4step_lora
```

### 8.3 原理

4-step LoRA 蒸馏的核心不是对真实视频做简单重建，而是条件分布匹配：

1. teacher 使用原始 40/50 步模型和 CFG，提供真实分布的 score/velocity。
2. fake score 模型学习当前 student 生成分布。
3. student/generator 只训练 LoRA 参数，在 4 步固定时间表上生成样本。
4. DMD loss 用 teacher 与 fake 的 score 差作为梯度方向，让 student 的 4 步生成分布逼近 teacher 的高步数分布。
5. CFG 蒸馏把 teacher 的 CFG 效果压进 student，所以推理时 `enable_cfg=false`，避免双重 CFG 导致发糊或过饱和。
6. 对 S2V 来说，条件分布是 `p(video | text, ref_image, audio, optional_pose)`；训练和推理必须保持音频帧对齐、参考图编码、motion token 注入、timestep/shift 完全一致。

## 9. 建议落地顺序

1. 先跑通现有 Wan2.2 MoE 4-step LoRA 推理，确认环境、模型路径、LoRA 本地路径、attention 算子可用。
2. 如果只需要现成 S2V LoRA 推理，先做推理侧最小支持：S2V step-distill scheduler + `lora_configs` 接入。
3. 如果要训练 S2V LoRA，优先实现 cached dataset，避免训练时重复跑 Wav2Vec/T5/VAE。
4. 再实现 trainable S2V model wrapper 和 `s2v_dmd_lora` trainer。
5. 最后做质量验证：4 步 no-CFG、音频同步、首帧一致性、pose 可控性、与 40 步 teacher 的成对对比。

