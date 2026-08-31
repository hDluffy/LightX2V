import importlib
import inspect
import math
import os
from glob import glob
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from loguru import logger
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file

from lightx2v_train.utils.registry import MODEL_REGISTER
from lightx2v_train.utils.utils import get_running_dtype

from .base import BaseModel


@dataclass
class WanS2VDenoiserInput:
    hidden_states: torch.Tensor


def _import_object(class_path):
    module_name, sep, object_name = class_path.replace(":", ".").rpartition(".")
    if not sep:
        raise ValueError(f"Invalid class path {class_path!r}; expected 'package.module:ClassName'.")
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


class GenericWanS2VTrainingAdapter(torch.nn.Module):
    """Generic adapter around an external trainable Wan2.2-S2V backbone."""

    @classmethod
    def from_pretrained(cls, model_path, torch_dtype=None, **kwargs):
        return cls(model_path=model_path, torch_dtype=torch_dtype, **kwargs)

    def __init__(
        self,
        model_path,
        backbone_class_path,
        torch_dtype=None,
        backbone_factory="from_pretrained",
        backbone_init_kwargs=None,
        forward_style="kwargs",
    ):
        super().__init__()
        if not backbone_class_path:
            raise RuntimeError("GenericWanS2VTrainingAdapter requires backbone_class_path in transformer_init_kwargs.")
        backbone_cls = _import_object(backbone_class_path)
        init_kwargs = dict(backbone_init_kwargs or {})
        if backbone_factory and hasattr(backbone_cls, backbone_factory):
            self.backbone = getattr(backbone_cls, backbone_factory)(model_path, torch_dtype=torch_dtype, **init_kwargs)
        else:
            init_kwargs.setdefault("model_path", model_path)
            self.backbone = backbone_cls(**init_kwargs)
        self.forward_style = forward_style

    def forward_lightx2v_s2v(self, hidden_states, timestep, context, seq_len, s2v):
        if hasattr(self.backbone, "forward_lightx2v_s2v"):
            return self.backbone.forward_lightx2v_s2v(
                hidden_states=hidden_states,
                timestep=timestep,
                context=context,
                seq_len=seq_len,
                s2v=s2v,
            )
        if self.forward_style == "kwargs":
            return self.backbone(hidden_states, t=timestep, context=context, seq_len=seq_len, s2v=s2v)
        if self.forward_style == "dict":
            return self.backbone(
                hidden_states=hidden_states,
                timestep=timestep,
                condition={"context": context, "s2v": s2v},
                seq_len=seq_len,
            )
        raise RuntimeError(f"Unsupported S2V adapter forward_style={self.forward_style!r}.")

    def forward(self, hidden_states, timestep, context, seq_len, s2v):
        return self.forward_lightx2v_s2v(hidden_states, timestep, context, seq_len, s2v)



class DiffSynthWanS2VBackbone(torch.nn.Module):
    """Trainable Wan2.2-S2V backbone backed by DiffSynth-Studio's WanVideo DiT.

    The LightX2V S2V cache stores target latents separately from reference
    latents. DiffSynth's S2V model function expects them concatenated with the
    reference latent in frame 0, so this adapter performs that shape bridge and
    returns only the target-frame velocity.
    """

    @classmethod
    def from_pretrained(cls, model_path, torch_dtype=None, **kwargs):
        return cls(model_path=model_path, torch_dtype=torch_dtype, **kwargs)

    def __init__(
        self,
        model_path,
        torch_dtype=None,
        model_paths=None,
        model_id_with_origin_paths=None,
        download_source=None,
        skip_download=None,
        load_device="cpu",
        redirect_common_files=False,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        strict_batch_size_one=True,
    ):
        super().__init__()
        self.model_path = model_path
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.load_device = load_device
        self.redirect_common_files = redirect_common_files
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_gradient_checkpointing_offload = bool(use_gradient_checkpointing_offload)
        self.strict_batch_size_one = bool(strict_batch_size_one)

        self._set_diffsynth_env(download_source, skip_download)
        WanVideoPipeline, ModelConfig, model_fn_wans2v = self._import_diffsynth()
        self.model_fn_wans2v = model_fn_wans2v
        self._model_fn_wans2v_accepts_add_last_motion = self._accepts_keyword(model_fn_wans2v, "add_last_motion")
        model_configs = self._build_model_configs(
            ModelConfig,
            model_path=model_path,
            model_paths=model_paths,
            model_id_with_origin_paths=model_id_with_origin_paths,
        )
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=self.torch_dtype,
            device=self.load_device,
            model_configs=model_configs,
            tokenizer_config=None,
            audio_processor_config=None,
        )
        if getattr(pipe, "dit", None) is None:
            raise RuntimeError(
                "DiffSynth WanVideoPipeline did not load a DiT. Set "
                "transformer_init_kwargs.model_paths to the Wan2.2-S2V "
                "diffusion_pytorch_model*.safetensors shards, or set "
                "model_id_with_origin_paths."
            )
        self.dit = pipe.dit
        self._make_audio_injection_autograd_safe()
        if hasattr(self.dit, "train"):
            self.dit.train()

    @staticmethod
    def _set_diffsynth_env(download_source, skip_download):
        if download_source:
            os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", str(download_source))
        if skip_download is not None:
            os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True" if skip_download else "False")

    @staticmethod
    def _import_diffsynth():
        try:
            from diffsynth import ModelConfig
            from diffsynth.pipelines.wan_video import WanVideoPipeline, model_fn_wans2v
        except ImportError as exc:
            raise ImportError(
                "DiffSynthWanS2VBackbone requires DiffSynth-Studio. Install it "
                "or add its repository to PYTHONPATH before training."
            ) from exc
        return WanVideoPipeline, ModelConfig, model_fn_wans2v

    @staticmethod
    def _accepts_keyword(function, keyword):
        try:
            parameters = inspect.signature(function).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(parameter.name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)

    def _make_audio_injection_autograd_safe(self):
        after_transformer_block = getattr(self.dit, "after_transformer_block", None)
        if after_transformer_block is None:
            return
        audio_injector = getattr(self.dit, "audio_injector", None)
        injected_block_ids = getattr(audio_injector, "injected_block_id", None)

        def checkpoint_safe_after_transformer_block(block_idx, hidden_states, *args, **kwargs):
            is_injection_block = injected_block_ids is None or block_idx in injected_block_ids
            if is_injection_block and torch.is_grad_enabled() and hidden_states.requires_grad:
                hidden_states = hidden_states.clone()
            return after_transformer_block(block_idx, hidden_states, *args, **kwargs)

        self.dit.after_transformer_block = checkpoint_safe_after_transformer_block

    def _build_model_configs(self, ModelConfig, model_path, model_paths=None, model_id_with_origin_paths=None):
        configs = []
        if model_id_with_origin_paths:
            for item in self._as_list(model_id_with_origin_paths):
                if isinstance(item, dict):
                    configs.append(ModelConfig(**item))
                    continue
                model_id, sep, origin_file_pattern = str(item).partition(":")
                if not sep:
                    raise ValueError(
                        "model_id_with_origin_paths entries must be "
                        "'model_id:origin_file_pattern' strings or ModelConfig kwargs."
                    )
                configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern))

        if model_paths is None and not configs:
            model_paths = self._infer_dit_paths(model_path)
        if model_paths:
            model_paths = self._expand_model_paths(model_paths)
            if isinstance(model_paths, (list, tuple)) and model_paths and all(not isinstance(item, dict) for item in model_paths):
                configs.append(ModelConfig(path=list(model_paths)))
            else:
                for item in self._as_list(model_paths):
                    if isinstance(item, dict):
                        configs.append(ModelConfig(**item))
                    else:
                        configs.append(ModelConfig(path=item))

        if not configs:
            raise RuntimeError("No DiffSynth model configs were built for Wan2.2-S2V DiT loading.")
        return configs

    @classmethod
    def _expand_model_paths(cls, model_paths):
        if isinstance(model_paths, str) and any(token in model_paths for token in "*?["):
            matches = sorted(glob(model_paths))
            return matches or model_paths
        if isinstance(model_paths, (list, tuple)):
            expanded = []
            for item in model_paths:
                if isinstance(item, str) and any(token in item for token in "*?["):
                    matches = sorted(glob(item))
                    expanded.extend(matches or [item])
                else:
                    expanded.append(item)
            return expanded
        return model_paths

    @classmethod
    def _infer_dit_paths(cls, model_path):
        path = Path(str(model_path))
        if any(token in str(path) for token in "*?["):
            matches = sorted(glob(str(path)))
            return matches or str(path)
        if path.is_file():
            return str(path)
        if not path.is_dir():
            return str(path)

        patterns = [
            "diffusion_pytorch_model*.safetensors",
            "transformer/diffusion_pytorch_model*.safetensors",
            "transformer/model*.safetensors",
        ]
        for pattern in patterns:
            matches = sorted(glob(str(path / pattern)))
            if matches:
                return matches if len(matches) > 1 else matches[0]
        return str(path)

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @property
    def blocks(self):
        return getattr(self.dit, "blocks", None)

    @property
    def in_dim(self):
        return getattr(self.dit, "in_dim", 16)

    def enable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = True
        if hasattr(self.dit, "enable_gradient_checkpointing"):
            self.dit.enable_gradient_checkpointing()

    def forward_lightx2v_s2v(self, hidden_states, timestep, context, seq_len=None, s2v=None):
        if s2v is None:
            raise RuntimeError("DiffSynthWanS2VBackbone.forward_lightx2v_s2v requires s2v conditioning.")

        hidden_states = self._ensure_5d(hidden_states, "hidden_states")
        batch_size = hidden_states.shape[0]
        if self.strict_batch_size_one and batch_size != 1:
            raise RuntimeError(
                "DiffSynth Wan2.2-S2V training currently expects per-rank batch_size=1. "
                "Set data.train.batch_size=1 or disable strict_batch_size_one after validating batch semantics."
            )

        ref_latents = self._ensure_5d(s2v["ref_latents"], "ref_latents", hidden_states)
        motion_latents = self._ensure_5d(s2v["motion_latents"], "motion_latents", hidden_states)
        cond_latents = self._ensure_5d(s2v["cond_latents"], "cond_latents", hidden_states)
        audio_input = self._ensure_audio(s2v["audio_input"], hidden_states)
        context = self._ensure_context(context, hidden_states)
        timestep = self._ensure_timestep(timestep, hidden_states)

        if ref_latents.shape[0] != batch_size:
            ref_latents = self._broadcast_batch(ref_latents, batch_size, "ref_latents")
        if motion_latents.shape[0] != batch_size:
            motion_latents = self._broadcast_batch(motion_latents, batch_size, "motion_latents")
        if cond_latents.shape[0] != batch_size:
            cond_latents = self._broadcast_batch(cond_latents, batch_size, "cond_latents")
        if audio_input.shape[0] != batch_size:
            audio_input = self._broadcast_batch(audio_input, batch_size, "audio_input")
        if context.shape[0] != batch_size:
            context = self._broadcast_batch(context, batch_size, "context")

        if batch_size == 1:
            return self._forward_one(
                hidden_states,
                timestep,
                context,
                ref_latents,
                motion_latents,
                cond_latents,
                audio_input,
                s2v,
            )

        outputs = []
        for idx in range(batch_size):
            outputs.append(
                self._forward_one(
                    hidden_states[idx : idx + 1],
                    timestep[idx : idx + 1],
                    context[idx : idx + 1],
                    ref_latents[idx : idx + 1],
                    motion_latents[idx : idx + 1],
                    cond_latents[idx : idx + 1],
                    audio_input[idx : idx + 1],
                    s2v,
                )
            )
        return torch.cat(outputs, dim=0)

    def forward(self, hidden_states, timestep, context, seq_len=None, s2v=None):
        return self.forward_lightx2v_s2v(hidden_states, timestep, context, seq_len, s2v)

    def _forward_one(self, hidden_states, timestep, context, ref_latents, motion_latents, cond_latents, audio_input, s2v):
        full_latents = torch.cat([ref_latents, hidden_states], dim=2)
        add_last_motion = int(s2v.get("add_last_motion", 2))
        model_kwargs = {
            "s2v_pose_latents": cond_latents,
            "motion_latents": motion_latents,
            "drop_motion_frames": bool(s2v.get("drop_motion_frames", False)),
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        if self._model_fn_wans2v_accepts_add_last_motion:
            model_kwargs["add_last_motion"] = add_last_motion
        elif add_last_motion != 2:
            raise RuntimeError(
                "The installed DiffSynth model_fn_wans2v does not support add_last_motion, "
                f"but the cached sample requests add_last_motion={add_last_motion}."
            )
        prediction = self.model_fn_wans2v(
            self.dit,
            full_latents,
            timestep,
            context,
            audio_input,
            **model_kwargs,
        )
        prediction = self._unwrap_prediction(prediction)
        if prediction.ndim == 4:
            prediction = prediction.unsqueeze(0)
        if prediction.shape[2] == hidden_states.shape[2] + ref_latents.shape[2]:
            prediction = prediction[:, :, ref_latents.shape[2] :]
        if prediction.shape != hidden_states.shape:
            raise RuntimeError(
                f"DiffSynth Wan2.2-S2V prediction shape {tuple(prediction.shape)} "
                f"does not match target latent shape {tuple(hidden_states.shape)}."
            )
        return prediction

    @staticmethod
    def _unwrap_prediction(prediction):
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        if isinstance(prediction, dict):
            prediction = prediction.get("sample", prediction.get("prediction", prediction.get("x", prediction)))
        return prediction

    @staticmethod
    def _broadcast_batch(tensor, batch_size, name):
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, *tensor.shape[1:])
        raise RuntimeError(f"{name} batch={tensor.shape[0]} does not match hidden_states batch={batch_size}.")

    @staticmethod
    def _ensure_5d(value, name, like=None):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if value.ndim == 4:
            value = value.unsqueeze(0)
        if value.ndim != 5:
            raise RuntimeError(f"{name} must be [B,C,F,H,W] or [C,F,H,W], got shape={tuple(value.shape)}.")
        if like is not None:
            value = value.to(device=like.device, dtype=like.dtype)
        return value

    @staticmethod
    def _ensure_audio(value, like):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4:
            raise RuntimeError(f"audio_input must be [B,L,D,F] or [L,D,F], got shape={tuple(value.shape)}.")
        return value.to(device=like.device, dtype=like.dtype)

    @staticmethod
    def _ensure_context(context, like):
        if not torch.is_tensor(context):
            context = torch.as_tensor(context)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context.ndim != 3:
            raise RuntimeError(f"context must be [B,L,D] or [L,D], got shape={tuple(context.shape)}.")
        return context.to(device=like.device, dtype=like.dtype)

    @staticmethod
    def _ensure_timestep(timestep, like):
        if not torch.is_tensor(timestep):
            timestep = torch.as_tensor(timestep)
        timestep = timestep.to(device=like.device, dtype=torch.float32).flatten()
        if timestep.numel() == 1 and like.shape[0] > 1:
            timestep = timestep.expand(like.shape[0])
        if timestep.numel() != like.shape[0]:
            raise RuntimeError(f"timestep batch={timestep.numel()} does not match hidden_states batch={like.shape[0]}.")
        return timestep


@MODEL_REGISTER("wan_s2v")
class WanS2VTrainModel(BaseModel):
    """Training wrapper for a trainable Wan2.2-S2V transformer.

    LightX2V's built-in Wan2.2-S2V inference model stores weights in custom
    non-nn.Module containers and runs under no_grad. Training therefore needs a
    trainable backbone class configured via model.transformer_class_path.
    """

    pipeline_cls = None

    def load_components(self, transformer_only=False, reference_model=None):
        model_config = self.config["model"]
        self.model_path = model_config["pretrained_model_name_or_path"]
        self.num_train_timesteps = self.config.get("scheduler", {}).get("num_train_timesteps", 1000)
        self.max_sequence_length = model_config.get("max_sequence_length", 512)
        self.vae_stride = tuple(model_config.get("vae_stride", (4, 8, 8)))
        self.patch_size = tuple(model_config.get("patch_size", (1, 2, 2)))
        self.vae_scale_factor_temporal = int(self.vae_stride[0])
        self.vae_scale_factor_spatial = int(self.vae_stride[1])
        default_param_dtype = "fp32" if self.config.get("training", {}).get("train_type") == "full" else model_config.get("running_dtype", "bf16")
        self.transformer_param_dtype = get_running_dtype(model_config.get("transformer_param_dtype", default_param_dtype))
        self.forward_style = model_config.get("forward_style", "lightx2v_s2v")
        self.context_as_list = bool(model_config.get("context_as_list", False))

        if transformer_only and reference_model is not None:
            self.max_sequence_length = reference_model.max_sequence_length
            self.vae_stride = reference_model.vae_stride
            self.patch_size = reference_model.patch_size
            self.vae_scale_factor_temporal = reference_model.vae_scale_factor_temporal
            self.vae_scale_factor_spatial = reference_model.vae_scale_factor_spatial

        self.transformer = self._load_transformer(model_config)
        self.transformer.to(self.device, dtype=self.transformer_param_dtype)

    def _load_transformer(self, model_config):
        class_path = model_config.get("transformer_class_path")
        if not class_path:
            module_name = model_config.get("transformer_module")
            class_name = model_config.get("transformer_class")
            if module_name and class_name:
                class_path = f"{module_name}:{class_name}"
        if not class_path:
            raise RuntimeError(
                "wan_s2v training requires a trainable S2V transformer. Set "
                "model.transformer_class_path to an nn.Module adapter, for example "
                "'my_pkg.wan_s2v_train:WanS2VTrainingAdapter'. The built-in "
                "lightx2v.models.networks.wan.s2v_model.WanS2VModel is inference-only "
                "and cannot be optimized with LoRA."
            )

        transformer_cls = _import_object(class_path)
        init_kwargs = dict(model_config.get("transformer_init_kwargs", {}))
        transformer_path = model_config.get("transformer_path", self.model_path)
        factory = model_config.get("transformer_factory", "from_pretrained")

        if factory and hasattr(transformer_cls, factory):
            transformer = getattr(transformer_cls, factory)(transformer_path, torch_dtype=self.transformer_param_dtype, **init_kwargs)
        else:
            init_kwargs.setdefault("model_path", transformer_path)
            transformer = transformer_cls(**init_kwargs)

        checkpoint_path = model_config.get("transformer_checkpoint")
        if checkpoint_path:
            self._load_transformer_checkpoint(transformer, checkpoint_path, strict=bool(model_config.get("transformer_checkpoint_strict", True)))
        return transformer

    def _load_transformer_checkpoint(self, transformer, checkpoint_path, strict=True):
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "model.safetensors")
        if checkpoint_path.endswith(".safetensors"):
            state = load_file(checkpoint_path)
        else:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            for key in ("state_dict", "model", "module"):
                if isinstance(state, dict) and key in state:
                    state = state[key]
                    break
        incompatible = transformer.load_state_dict(state, strict=strict)
        if not strict:
            if incompatible.missing_keys:
                logger.warning("Missing keys when loading S2V transformer checkpoint: {}", incompatible.missing_keys)
            if incompatible.unexpected_keys:
                logger.warning("Unexpected keys when loading S2V transformer checkpoint: {}", incompatible.unexpected_keys)

    def denoiser_module(self):
        if self.transformer is None:
            raise RuntimeError("Wan S2V transformer is not loaded.")
        return self.transformer

    def transformer_forward_context(self):
        if self.device.type == "cuda" and self.running_dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type="cuda", dtype=self.running_dtype)
        return nullcontext()

    def add_lora(self, rank, alpha, target_modules):
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        try:
            self.transformer = inject_adapter_in_model(lora_config, self.transformer, adapter_name="default")
        except TypeError:
            self.transformer = inject_adapter_in_model(lora_config, self.transformer)

    def encode_to_latent(self, sample):
        latent = sample.get("latent", sample.get("target_latent"))
        if latent is None:
            raise RuntimeError("wan_s2v DMD uses cached conditioning and latent_shape; target latent is optional and was not provided.")
        latent = latent.to(device=self.device, dtype=self.running_dtype)
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        return latent

    def encode_condition(self, sample, unconditional=False):
        context_key = "context_null" if unconditional else "context"
        context = sample.get(context_key)
        if context is None and unconditional:
            context = sample.get("negative_prompt_embed")
        if context is None and not unconditional:
            context = sample.get("prompt_embed")
        if context is None:
            missing = "context_null/negative_prompt_embed" if unconditional else "context/prompt_embed"
            raise RuntimeError(f"wan_s2v cached sample is missing {missing}. Build S2V caches before training.")

        condition = {
            "context": self._tensor(context, dtype=self.running_dtype),
            "ref_latents": self._tensor(self._required(sample, "ref_latents"), dtype=self.running_dtype),
            "motion_latents": self._tensor(self._required(sample, "motion_latents"), dtype=self.running_dtype),
            "cond_latents": self._tensor(self._required(sample, "cond_latents"), dtype=self.running_dtype),
            "audio_input": self._tensor(self._required_any(sample, ("audio_input", "audio_emb")), dtype=self.running_dtype),
            "motion_frames": self._motion_frames(sample),
            "drop_motion_frames": self._as_bool(sample.get("drop_motion_frames", False)),
            "add_last_motion": int(self._scalar(sample.get("add_last_motion", 2))),
        }
        if unconditional:
            condition["audio_input"] = torch.zeros_like(condition["audio_input"])
        if "height" in sample:
            condition["height"] = int(self._scalar(sample["height"]))
        if "width" in sample:
            condition["width"] = int(self._scalar(sample["width"]))
        return condition

    def prepare_denoiser_input(self, noisy_latent, condition=None):
        return WanS2VDenoiserInput(hidden_states=noisy_latent)

    def denoise(self, denoiser_input, timestep_or_sigma, condition):
        timestep = timestep_or_sigma.float() * self.num_train_timesteps
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(device=self.device)

        hidden_states = denoiser_input.hidden_states.to(device=self.device, dtype=self.running_dtype)
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.unsqueeze(0)

        seq_len = self._sequence_length(hidden_states)
        with self.transformer_forward_context():
            return self._forward_transformer(hidden_states, timestep, condition, seq_len)

    def _forward_transformer(self, hidden_states, timestep, condition, seq_len):
        denoiser = self.denoiser_module()
        context = self._context_for_forward(condition["context"])
        s2v = {
            "ref_latents": condition["ref_latents"],
            "motion_latents": condition["motion_latents"],
            "cond_latents": condition["cond_latents"],
            "audio_input": condition["audio_input"],
            "motion_frames": condition["motion_frames"],
            "drop_motion_frames": condition["drop_motion_frames"],
            "add_last_motion": condition["add_last_motion"],
        }

        if hasattr(denoiser, "forward_lightx2v_s2v"):
            if self.is_fsdp2_wrapped():
                return denoiser(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    context=context,
                    seq_len=seq_len,
                    s2v=s2v,
                )
            return denoiser.forward_lightx2v_s2v(
                hidden_states=hidden_states,
                timestep=timestep,
                context=context,
                seq_len=seq_len,
                s2v=s2v,
            )

        if self.forward_style == "kwargs":
            return denoiser(hidden_states, t=timestep, context=context, seq_len=seq_len, s2v=s2v)
        if self.forward_style == "dict":
            return denoiser(hidden_states=hidden_states, timestep=timestep, condition=condition, seq_len=seq_len)
        raise RuntimeError(
            "The configured wan_s2v transformer does not expose forward_lightx2v_s2v(). "
            "Either implement that adapter method or set model.forward_style to 'kwargs'/'dict' "
            "for a compatible forward signature."
        )

    def postprocess_denoiser_output(self, prediction, denoiser_input):
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        if isinstance(prediction, dict):
            prediction = prediction.get("sample", prediction.get("prediction", prediction.get("x", prediction)))
        return prediction

    def prepare_infer_latents(self, height, width, generator=None):
        infer_frames = self.config.get("inference", {}).get("infer_frames", self.config.get("inference", {}).get("num_frames", 80))
        motion_frames = self.config.get("inference", {}).get("motion_frames", 73)
        lat_motion_frames = (int(motion_frames) + 3) // 4
        latent_frames = (int(infer_frames) + 3 + int(motion_frames)) // 4 - lat_motion_frames
        shape = (1, self._latent_channels(), latent_frames, height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial)
        return torch.randn(shape, generator=generator, device=self.device, dtype=self.running_dtype)

    def dmd_latent_shape(self, batch_size, height, width):
        infer_config = self.config.get("inference", {})
        infer_frames = infer_config.get("infer_frames", 80)
        motion_frames = infer_config.get("motion_frames", 73)
        lat_motion_frames = (int(motion_frames) + 3) // 4
        latent_frames = (int(infer_frames) + 3 + int(motion_frames)) // 4 - lat_motion_frames
        return (batch_size, self._latent_channels(), latent_frames, int(height) // self.vae_scale_factor_spatial, int(width) // self.vae_scale_factor_spatial)

    def decode_latent(self, latent):
        raise NotImplementedError("wan_s2v training wrapper does not include VAE decoding. Use LightX2V S2V inference for validation.")

    def assemble_pipeline(self, scheduler=None):
        raise NotImplementedError("wan_s2v uses LightX2V inference scripts instead of a training-side pipeline.")

    def get_pipeline_infer_kwargs(self, infer_config):
        return {}

    def fsdp2_shard_plan(self, fsdp_config):
        reshard_config = fsdp_config.get("reshard_after_forward", {"root_reshard": False, "block_reshard": True})
        blocks = getattr(self.transformer, "blocks", None)
        if blocks is None:
            blocks = getattr(self.transformer, "layers", None)
        if blocks is None:
            return [{"module": self.transformer, "reshard_after_forward": reshard_config.get("root_reshard", False)}]
        return [
            {"modules": blocks, "reshard_after_forward": reshard_config.get("block_reshard", True)},
            {"module": self.transformer, "reshard_after_forward": reshard_config.get("root_reshard", False)},
        ]

    def _sequence_length(self, latent):
        latent_frames, latent_height, latent_width = latent.shape[-3:]
        patch_t, patch_h, patch_w = self.patch_size
        seq_len = (latent_frames // patch_t) * (latent_height // patch_h) * (latent_width // patch_w)
        return math.ceil(seq_len)

    def _context_for_forward(self, context):
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if not self.context_as_list:
            return context
        return [item for item in context.unbind(0)]

    def _latent_channels(self):
        if self.transformer is not None:
            for attr in ("in_dim", "in_channels", "num_channels_latents"):
                value = getattr(self.transformer, attr, None)
                if value is not None:
                    return int(value)
            config = getattr(self.transformer, "config", None)
            if config is not None:
                for attr in ("in_dim", "in_channels", "num_channels_latents"):
                    value = getattr(config, attr, None)
                    if value is not None:
                        return int(value)
        return int(self.config["model"].get("latent_channels", 16))

    def _tensor(self, value, dtype=None):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return value.to(device=self.device, dtype=dtype)

    @staticmethod
    def _required(sample, key):
        if key not in sample:
            raise RuntimeError(f"wan_s2v cached sample is missing {key!r}.")
        return sample[key]

    @staticmethod
    def _required_any(sample, keys):
        for key in keys:
            if key in sample:
                return sample[key]
        raise RuntimeError(f"wan_s2v cached sample is missing one of {keys!r}.")

    def _motion_frames(self, sample):
        value = sample.get("motion_frames")
        if value is None:
            return (int(self.config.get("inference", {}).get("motion_frames", 73)), int((self.config.get("inference", {}).get("motion_frames", 73) + 3) // 4))
        if torch.is_tensor(value):
            value = value.detach().cpu().flatten().tolist()
        return tuple(int(v) for v in value[:2])

    @staticmethod
    def _scalar(value):
        if torch.is_tensor(value):
            return value.detach().cpu().flatten()[0].item()
        if isinstance(value, (list, tuple)):
            return value[0]
        return value

    @classmethod
    def _as_bool(cls, value):
        return bool(cls._scalar(value))
