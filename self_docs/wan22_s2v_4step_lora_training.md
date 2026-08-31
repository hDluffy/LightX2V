# Wan2.2-S2V 4-Step LoRA 蒸馏训练说明

## 当前实现范围

本次已补齐 Wan2.2-S2V 4-step LoRA 蒸馏训练的 LightX2V 训练入口、缓存数据集、训练配置、缓存构造脚本和推理验证脚本：

- 训练模型注册：`lightx2v_train/lightx2v_train/model_zoo/wan_s2v.py`
- 训练器注册：`lightx2v_train/lightx2v_train/trainers/s2v_dmd.py`
- S2V cached dataset：`lightx2v_train/lightx2v_train/data/s2v_dataset.py`
- 训练配置：`lightx2v_train/configs/train/dmd/wan22_s2v_4step_lora.yaml`
- 缓存构造脚本：`lightx2v_train/scripts/build_wan22_s2v_cache.py`
- 缓存构造 shell：`lightx2v_train/scripts/build_wan22_s2v_cache.sh`
- 训练 shell：`lightx2v_train/scripts/run_wan22_s2v_4step_lora.sh`
- 4-step S2V LoRA 推理 scheduler：`lightx2v/models/schedulers/wan/s2v/step_distill_scheduler.py`
- 4-step S2V LoRA 推理配置：`configs/wan22/wan_s2v_distill_lora_4step.json`
- 4-step S2V LoRA 推理 shell：`scripts/wan22/distill/run_wan22_s2v_distill_lora_4step.sh`

注意：仓库内原有 `lightx2v.models.networks.wan.s2v_model.WanS2VModel` 是推理优化实现，不是可训练 `torch.nn.Module` 参数树，且关键路径包含 `@torch.no_grad()` 和自定义权重容器。因此训练配置要求 `model.transformer_class_path` 指向一个训练版 S2V transformer adapter。

## Conda 环境

沿用仓库训练环境，建议从已有 `requirements` 安装后补齐训练依赖：

```bash
conda create -n lightx2v-train python=3.10 -y
conda activate lightx2v-train
cd /data/hjq/LightX2V

pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install peft safetensors omegaconf loguru imageio decord
```

若使用自定义 Wan2.2-S2V 训练 adapter，将 adapter 包加入 `PYTHONPATH`。

## 训练版 S2V Backbone 接口

编辑 `lightx2v_train/configs/train/dmd/wan22_s2v_4step_lora.yaml`：

```yaml
model:
    name: wan_s2v
    pretrained_model_name_or_path: /path/to/Wan2.2-S2V-14B
    transformer_class_path: lightx2v_train.model_zoo.wan_s2v:GenericWanS2VTrainingAdapter
    transformer_factory: from_pretrained
    transformer_init_kwargs:
        backbone_class_path: your_package.wan_s2v_train:WanS2VBackbone
        backbone_factory: from_pretrained
        forward_style: kwargs
```

推荐真实 backbone 或 adapter 暴露：

```python
forward_lightx2v_s2v(
    hidden_states,  # [B, C, F, H, W]
    timestep,       # [B] or [1], scaled to [0, 1000]
    context,        # [B, L, D]
    seq_len,
    s2v,            # ref_latents/motion_latents/cond_latents/audio_input/motion_frames
)
```

如果 adapter 的 forward 签名不同，可设置：

```yaml
model:
    transformer_init_kwargs:
        forward_style: kwargs  # calls backbone(hidden_states, t=..., context=..., seq_len=..., s2v=...)
```

## 数据构造

准备 JSONL/CSV/JSON metadata，默认字段：

```json
{"prompt":"...", "image_path":"/path/ref.jpg", "audio_path":"/path/talk.wav", "negative_prompt":" ", "src_pose_path":""}
```

构造缓存：

```bash
cd /data/hjq/LightX2V/lightx2v_train
MODEL_PATH=/path/to/Wan2.2-S2V-14B \
CONFIG_JSON=../configs/wan22/wan_s2v.json \
METADATA_PATH=/path/to/train.jsonl \
OUTPUT_DIR=./data_cache/wan22_s2v/train \
bash scripts/build_wan22_s2v_cache.sh --overwrite
```

缓存 `.pt` 主要字段：

- `context` / `context_null`
- `ref_latents`
- `motion_latents`
- `cond_latents`
- `audio_input`
- `motion_frames`
- `latent_shape`
- `drop_motion_frames`
- `add_last_motion`

## 训练参数

默认 4-step 蒸馏 timestep：

```yaml
denoising_step_list: [1000, 750, 500, 250]
```

默认 LoRA：

```yaml
lora:
    rank: 64
    alpha: 64
    target_modules: [q, k, v, o, ffn.0, ffn.2]
```

DMD 参数：

- student learning rate：`2e-6`
- fake learning rate：`4e-7`
- `fake_update_ratio: 5`
- teacher CFG：`guidance_scale: 4.5`
- `timestep_shift: 5.0`
- `max_grad_norm: 1.0`

启动训练：

```bash
cd /data/hjq/LightX2V/lightx2v_train
CONFIG=configs/train/dmd/wan22_s2v_4step_lora.yaml \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash scripts/run_wan22_s2v_4step_lora.sh
```

## 推理验证

训练后修改：

```json
"lora_configs": [
  {
    "path": "/path/to/output_train/wan22_s2v_4step_lora/checkpoint-000001000/pytorch_lora_weights.safetensors",
    "strength": 1.0
  }
]
```

然后运行：

```bash
cd /data/hjq/LightX2V
bash scripts/wan22/distill/run_wan22_s2v_distill_lora_4step.sh
```

该推理路径使用 `WanS2VStepDistillScheduler`，`infer_steps=4`，并在 `WanS2VRunner.load_transformer()` 中加载 `lora_configs`。
