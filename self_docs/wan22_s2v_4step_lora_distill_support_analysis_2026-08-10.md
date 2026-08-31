# Wan2.2-S2V 4-Step LoRA 蒸馏支持分析

日期：2026-08-10

## 结论

当前仓库已经具备 Wan2.2-S2V 4-step LoRA 推理支持：可以在 `model_cls=wan2.2_s2v` 下通过 `WanS2VStepDistillScheduler`、`denoising_step_list=[1000,750,500,250]` 和 `lora_configs` 加载单个 S2V LoRA，并以 4 步完成 S2V 推理。

当前仓库不具备开箱即用的 Wan2.2-S2V 4-step LoRA 蒸馏训练闭环。训练侧已经补了注册、缓存数据集、DMD LoRA trainer、缓存构造脚本和 YAML 配置，但默认 YAML 仍要求用户提供真实可训练的 S2V backbone：`your_package.wan_s2v_train:WanS2VBackbone`。仓库内置的 `lightx2v.models.networks.wan.s2v_model.WanS2VModel` 是推理优化实现，不是可直接反传训练的 `nn.Module` 参数树。

因此建议对外表述为：

- 已支持：使用已有 Wan2.2-S2V 4-step LoRA 权重做 4 步推理验证。
- 部分支持：S2V 4-step LoRA DMD 蒸馏训练编排、缓存和导出路径。
- 未完全支持：仓库内置完整可训练 Wan2.2-S2V backbone；需要接入官方或自研训练版 S2V transformer adapter。

## 证据整理

### 推理侧支持点

1. `WanS2VRunner` 已注册为 `wan2.2_s2v`，并在 `init_scheduler()` 中根据 `scheduler_type == "WanS2VStepDistillScheduler"` 或 `denoising_step_list` 切换到 S2V 4-step scheduler。

   位置：`lightx2v/models/runners/wan/wan_s2v_runner.py:58`、`:67`、`:68`

2. `WanS2VRunner.load_transformer()` 已读取 `lora_configs`，存在 LoRA 时调用通用 `build_wan_model_with_lora(..., model_type="wan2.2_s2v")`。

   位置：`lightx2v/models/runners/wan/wan_s2v_runner.py:88`、`:90`、`:93`

3. 新增的 `WanS2VStepDistillScheduler` 固定读取 `denoising_step_list`，并把推理步数设为该列表长度。

   位置：`lightx2v/models/schedulers/wan/s2v/step_distill_scheduler.py:7`、`:12`、`:13`

4. 4-step scheduler 通过 `sample_shift` 把线性 sigma 变换成 Wan 系列 flow-matching sigma，再用 `num_train_timesteps - step` 映射 `[1000,750,500,250]` 到实际索引。

   位置：`lightx2v/models/schedulers/wan/s2v/step_distill_scheduler.py:28`、`:33`、`:34`、`:35`

5. 推理配置已存在：`configs/wan22/wan_s2v_distill_lora_4step.json`，核心字段包括 `infer_steps=4`、`scheduler_type`、`sample_shift=5.0`、`enable_cfg=false`、`denoising_step_list` 和 `lora_configs`。

   位置：`configs/wan22/wan_s2v_distill_lora_4step.json:2`、`:23`、`:29`

6. 推理脚本已存在：`scripts/wan22/distill/run_wan22_s2v_distill_lora_4step.sh`，调用 `python -m lightx2v.infer --model_cls wan2.2_s2v --task s2v`。

   位置：`scripts/wan22/distill/run_wan22_s2v_distill_lora_4step.sh:3`

### 训练侧支持点与限制

1. 训练模型已注册为 `wan_s2v`，但实现说明里明确写了训练需要外部可训练 backbone，不能直接使用推理版 `WanS2VModel`。

   位置：`lightx2v_train/lightx2v_train/model_zoo/wan_s2v.py:81`、`:86`

2. `GenericWanS2VTrainingAdapter` 支持从 `backbone_class_path` 动态导入外部 `nn.Module`，并转发 `forward_lightx2v_s2v(...)` 或兼容 `kwargs` / `dict` 风格 forward。

   位置：`lightx2v_train/lightx2v_train/model_zoo/wan_s2v.py:31`、`:41`、`:49`、`:50`

3. 默认训练配置仍是占位 adapter：`backbone_class_path: your_package.wan_s2v_train:WanS2VBackbone`，运行前必须替换。

   位置：`lightx2v_train/configs/train/dmd/wan22_s2v_4step_lora.yaml:5`、`:9`

4. `S2VDmdLoraTrainer` 已覆盖 `VideoDmdTrainer` 的限制，允许 `model.name=wan_s2v` 和 `training.train_type=lora`。

   位置：`lightx2v_train/lightx2v_train/trainers/s2v_dmd.py:9`、`:11`、`:12`

5. S2V cached dataset 已注册为 `wan_s2v_cached_dataset`，要求缓存中有 `context/context_null`、`ref_latents`、`motion_latents`、`cond_latents`、`audio_input/audio_emb`，并能提供或推断 `latent_shape`。

   位置：`lightx2v_train/lightx2v_train/data/s2v_dataset.py:10`、`:13`、`:15`、`:138`

6. 缓存构造脚本复用推理 runner 编码参考图、文本、音频、pose 条件，并保存训练所需字段。

   位置：`lightx2v_train/scripts/build_wan22_s2v_cache.py:122`、`:126`、`:151`、`:161`、`:163`、`:167`

7. LoRA 保存与推理读取方向基本匹配：训练侧保存 `pytorch_lora_weights.safetensors`；推理侧加载时会把 `lora_A/lora_B` 兼容为 `lora_down/lora_up`。

   位置：`lightx2v_train/lightx2v_train/model_zoo/base.py:204`、`lightx2v/models/networks/base_model.py:343`、`:346`

### 静态验证

已对相关新增/修改 Python 文件执行静态编译，命令如下，结果通过：

```bash
python -m py_compile \
  lightx2v/models/schedulers/wan/s2v/step_distill_scheduler.py \
  lightx2v/models/runners/wan/wan_s2v_runner.py \
  lightx2v_train/lightx2v_train/model_zoo/wan_s2v.py \
  lightx2v_train/lightx2v_train/trainers/s2v_dmd.py \
  lightx2v_train/lightx2v_train/data/s2v_dataset.py \
  lightx2v_train/scripts/build_wan22_s2v_cache.py
```

未做真实训练/推理验证，原因是本仓库没有本地 Wan2.2-S2V-14B 权重路径和真实 S2V 4-step LoRA 权重路径，训练配置里的 backbone 也是占位值。

## 4-Step LoRA 推理执行流程

### 1. 准备权重

需要三类路径：

- `LightX2V` 仓库路径，例如 `/data/hjq/LightX2V`。
- Wan2.2-S2V-14B 原始模型目录，例如 `/path/to/Wan2.2-S2V-14B`。
- 已训练好的 S2V 4-step LoRA 权重，例如 `/path/to/output_train/wan22_s2v_4step_lora/checkpoint-000001000/pytorch_lora_weights.safetensors`。

### 2. 修改推理配置

编辑 `configs/wan22/wan_s2v_distill_lora_4step.json`：

```json
{
  "scheduler_type": "WanS2VStepDistillScheduler",
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
  "use_image_encoder": false,
  "denoising_step_list": [1000, 750, 500, 250],
  "lora_configs": [
    {
      "path": "/abs/path/to/pytorch_lora_weights.safetensors",
      "strength": 1.0
    }
  ]
}
```

注意：

- `path` 建议使用绝对路径。
- S2V 当前是单模型 LoRA 配置，不是 Wan2.2 MoE T2V/I2V 那种 high/low 双 LoRA。
- 4-step 蒸馏通常关闭 CFG，所以 `enable_cfg=false`、`sample_guide_scale=1.0`。
- 如果使用已有 40-step S2V 配置，默认是 `sample_shift=3`、`sample_guide_scale=4.5`、`enable_cfg=true`，不等价于 4-step distilled LoRA 推理。

### 3. 单卡运行

```bash
cd /data/hjq/LightX2V

lightx2v_path=/data/hjq/LightX2V
model_path=/path/to/Wan2.2-S2V-14B

export CUDA_VISIBLE_DEVICES=0
source ${lightx2v_path}/scripts/base/base.sh

python -m lightx2v.infer \
  --model_cls wan2.2_s2v \
  --task s2v \
  --model_path "${model_path}" \
  --config_json ${lightx2v_path}/configs/wan22/wan_s2v_distill_lora_4step.json \
  --prompt "A person speaks naturally to the camera with subtle head motion." \
  --negative_prompt "画面模糊，最差质量，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
  --image_path /path/to/ref_image.jpg \
  --audio_path /path/to/audio.wav \
  --save_result_path ${lightx2v_path}/save_results/output_lightx2v_wan22_s2v_4step_lora.mp4
```

也可以直接修改并运行：

```bash
bash scripts/wan22/distill/run_wan22_s2v_distill_lora_4step.sh
```

### 4. 带 pose 条件

`WanS2VRunner` 会读取 `src_pose_path`，如果文件存在则走 `load_pose_cond()`，否则忽略 pose 条件。命令中增加：

```bash
  --src_pose_path /path/to/pose.mp4
```

### 5. 结果检查

重点检查：

- 日志中出现 `Using WanS2VStepDistillScheduler`。
- 日志中出现 LoRA applied 或对应 LoRA 加载成功信息。
- 生成步数为 `1 / 4` 到 `4 / 4`。
- 输出 mp4 已写入 `save_result_path`，且如果输入音频存在，会尝试 mux 音频到视频。

## 训练执行流程

该流程只适用于已经准备好真实可训练 S2V backbone adapter 的情况。

### 1. 准备训练版 backbone

训练配置里的：

```yaml
model:
  transformer_class_path: lightx2v_train.model_zoo.wan_s2v:GenericWanS2VTrainingAdapter
  transformer_init_kwargs:
    backbone_class_path: your_package.wan_s2v_train:WanS2VBackbone
```

必须替换为实际可导入的训练版 `nn.Module`。推荐 adapter 暴露：

```python
forward_lightx2v_s2v(
    hidden_states,  # [B, C, F, H, W]
    timestep,       # [B] or [1], scaled to [0, 1000]
    context,        # [B, L, D] or compatible list
    seq_len,
    s2v,            # ref_latents/motion_latents/cond_latents/audio_input/motion_frames
)
```

若 adapter 不提供该方法，可按配置使用 `forward_style: kwargs`，此时会调用：

```python
backbone(hidden_states, t=timestep, context=context, seq_len=seq_len, s2v=s2v)
```

### 2. 准备训练 metadata

JSONL 示例：

```jsonl
{"prompt":"a person is speaking to camera", "image_path":"/data/images/000001.jpg", "audio_path":"/data/audio/000001.wav", "negative_prompt":" ", "src_pose_path":""}
{"prompt":"a person is singing", "image_path":"/data/images/000002.jpg", "audio_path":"/data/audio/000002.wav", "negative_prompt":" ", "src_pose_path":"/data/pose/000002.mp4"}
```

字段说明：

- `prompt`：文本条件。
- `image_path`：参考人像或参考图。
- `audio_path`：驱动音频。
- `negative_prompt`：负向 prompt，可为空格。
- `src_pose_path`：可选 pose 视频。

### 3. 构造 S2V 条件缓存

缓存构造会使用推理 runner 计算并保存文本、音频、参考图、motion/pose latents，减少训练时重复编码成本。

```bash
cd /data/hjq/LightX2V/lightx2v_train

MODEL_PATH=/path/to/Wan2.2-S2V-14B \
CONFIG_JSON=../configs/wan22/wan_s2v.json \
METADATA_PATH=/path/to/wan22_s2v_train.jsonl \
OUTPUT_DIR=./data_cache/wan22_s2v/train \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/build_wan22_s2v_cache.sh --overwrite
```

建议为验证集单独构造：

```bash
MODEL_PATH=/path/to/Wan2.2-S2V-14B \
CONFIG_JSON=../configs/wan22/wan_s2v.json \
METADATA_PATH=/path/to/wan22_s2v_val.jsonl \
OUTPUT_DIR=./data_cache/wan22_s2v/val \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/build_wan22_s2v_cache.sh --overwrite
```

缓存 `.pt` 核心字段：

- `context`
- `context_null`
- `ref_latents`
- `motion_latents`
- `cond_latents`
- `audio_input`
- `motion_frames`
- `latent_shape`
- `drop_motion_frames`
- `add_last_motion`

### 4. 修改训练 YAML

编辑 `lightx2v_train/configs/train/dmd/wan22_s2v_4step_lora.yaml`：

```yaml
model:
  pretrained_model_name_or_path: /path/to/Wan2.2-S2V-14B
  transformer_class_path: lightx2v_train.model_zoo.wan_s2v:GenericWanS2VTrainingAdapter
  transformer_init_kwargs:
    backbone_class_path: your_real_package.wan_s2v_train:WanS2VBackbone
    backbone_factory: from_pretrained
    backbone_init_kwargs: {}
    forward_style: kwargs
  fake:
    name: wan_s2v
    pretrained_model_name_or_path: /path/to/Wan2.2-S2V-14B
  teacher:
    name: wan_s2v
    pretrained_model_name_or_path: /path/to/Wan2.2-S2V-14B

data:
  train:
    cache_path:
      - /data/hjq/LightX2V/lightx2v_train/data_cache/wan22_s2v/train
  val:
    cache_path:
      - /data/hjq/LightX2V/lightx2v_train/data_cache/wan22_s2v/val

training:
  method: s2v_dmd_lora
  train_type: lora
  dmd:
    num_inference_steps: 4
    denoising_step_list: [1000, 750, 500, 250]
    timestep_shift: 5.0
  output_dir: ./output_train/wan22_s2v_4step_lora
```

### 5. 启动训练

单卡 smoke test：

```bash
cd /data/hjq/LightX2V/lightx2v_train

CONFIG=configs/train/dmd/wan22_s2v_4step_lora.yaml \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
bash scripts/run_wan22_s2v_4step_lora.sh
```

多卡：

```bash
CONFIG=configs/train/dmd/wan22_s2v_4step_lora.yaml \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash scripts/run_wan22_s2v_4step_lora.sh
```

### 6. 训练产物

LoRA checkpoint 默认保存为：

```text
lightx2v_train/output_train/wan22_s2v_4step_lora/checkpoint-000001000/pytorch_lora_weights.safetensors
```

把该路径填回 `configs/wan22/wan_s2v_distill_lora_4step.json` 的 `lora_configs[0].path` 后，按上面的 4-step 推理流程验证。

## 原理说明

### 1. 4-step step distill 的采样原理

Wan 系列使用 flow matching 形式的速度预测。普通 S2V 默认使用 40 步 UniPC 调度：每一步根据当前 latent、timestep、文本/音频/参考图/pose 条件预测 velocity，再由 scheduler 做数值积分。

4-step distilled LoRA 的目标是让模型在很少的固定 timestep 上直接预测足够好的 velocity。这里固定选择：

```text
[1000, 750, 500, 250]
```

这些 step 会经 `sample_shift=5.0` 映射成推理用 sigma。每一步的更新形式是：

```text
x0_pred = x_t - sigma_t * v_theta(x_t, t, cond)
x_next = x0_pred + sigma_next * v_theta(x_t, t, cond)
```

最后一步 `sigma_next=0`，所以结果就是预测的 clean latent。实现对应 `WanS2VStepDistillScheduler.step_post()`。

### 2. 为什么 4-step 推理一般关闭 CFG

常规 CFG 会在每一步分别跑 conditional 和 unconditional，再做：

```text
v = v_uncond + scale * (v_cond - v_uncond)
```

这会增加一倍 denoiser 计算，并且训练出的 4-step distilled LoRA 通常已经把高 CFG teacher 的效果压缩进学生模型。因此 4-step 配置使用 `enable_cfg=false`，让每步只跑一次 denoiser。

### 3. LoRA 蒸馏训练的核心

训练侧采用 DMD 类方法时通常包含三个角色：

- student：当前要训练的 LoRA，负责 4-step 反推轨迹。
- fake：跟随 student 的辅助模型，用于估计生成分布的 score。
- teacher：冻结的原模型或高质量模型，用较强 CFG 给出参考 score。

训练会先让 student 从噪声走若干 4-step 轨迹，得到中间样本或 `x0`；再把样本重新加噪，让 fake 与 teacher 在同一 noisy latent 上预测 velocity。DMD loss 本质上用 teacher 与 fake 的差异构造梯度，推动 student 的短步生成分布靠近 teacher 分布。

### 4. S2V 条件为什么要缓存

S2V 条件比 T2V/I2V 更复杂，除了文本外还有：

- 参考图 VAE latent：控制主体外观。
- motion latents：承接上一段或首段 motion 条件。
- audio embedding：按 `infer_frames` 切片，驱动口型/动作。
- pose cond latents：可选，提供姿态引导。

这些编码成本高、格式细节多。训练侧先用推理 runner 构造 `.pt` 缓存，可以让 DMD 训练集中在 denoiser/LoRA 参数更新上，也避免训练版 backbone 和推理版条件预处理不一致。

### 5. 当前训练缺口的本质

推理版 `WanS2VModel` 是 LightX2V 的高性能推理栈：权重放在 `WeightModule` 容器中，核心调用路径带 `@torch.no_grad()`，并通过自定义 infer module 分段执行。它适合低显存、高性能推理，但不适合直接被 PEFT 注入 LoRA 并反传优化。

所以当前训练配置设计成 adapter 方式：LightX2V 提供训练编排、缓存、DMD loss、LoRA 保存；真实可训练 S2V transformer 由外部 adapter 接入。只要 adapter 的 forward 与缓存字段匹配，训练流程就能跑通。

## 风险与检查项

- LoRA target modules 必须和训练版 backbone 的 `named_modules()` 对齐；默认 `[q,k,v,o,ffn.0,ffn.2]` 只是起点。
- 推理版 LoRA key 必须能被 LightX2V 的 `LoRALoader` 匹配到 `original_weight_dict`；训练后首次验证应打开日志确认应用成功。
- 如果 backbone 输出不是 velocity 或形状不是 `[B,C,F,H,W]`，需要在 `postprocess_denoiser_output()` 或 adapter 内转换。
- 如果使用 pose 条件，训练缓存和推理验证都应使用相同 `infer_frames`、`motion_frames`、`drop_first_motion` 和 `target_fps`。
- 真实训练前建议先用 1-2 条缓存、`max_train_iters=1` 做 smoke test，再扩大数据和步数。

