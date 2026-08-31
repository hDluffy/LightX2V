import torch

from lightx2v_train.utils.registry import TRAINER_REGISTER

from .dmd import VideoDmdTrainer


@TRAINER_REGISTER("s2v_dmd_lora")
class S2VDmdLoraTrainer(VideoDmdTrainer):
    trainer_name = "s2v_dmd_lora"
    allowed_model_names = {"wan_s2v"}
    allowed_train_types = {"lora"}

    def _encode_conditions(self, sample):
        with torch.no_grad():
            condition = self.model.encode_condition(sample, unconditional=False)
            negative_condition = None
            if self.guidance_scale != 0:
                negative_condition = self.model.encode_condition(sample, unconditional=True)
        return condition, negative_condition

    def _latent_shape(self, sample):
        prompt = sample.get("prompt", "")
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        configured_shape = self.dmd_config.get("image_or_video_shape", self.training_config.get("image_or_video_shape"))
        if configured_shape is not None:
            shape = list(configured_shape)
            shape[0] = batch_size
            return tuple(int(dim) for dim in shape)

        if "latent_shape" in sample:
            shape = sample["latent_shape"]
            if torch.is_tensor(shape):
                if shape.ndim > 1:
                    shape = shape[0]
                shape = shape.detach().cpu().tolist()
            shape = [int(dim) for dim in shape]
            if len(shape) == 4:
                shape = [batch_size, *shape]
            else:
                shape[0] = batch_size
            return tuple(shape)

        if "latent" in sample:
            latent = sample["latent"]
            return tuple(int(dim) for dim in (latent.shape if latent.ndim == 5 else (batch_size, *latent.shape)))

        height = self._sample_int(sample, "height", self.config.get("inference", {}).get("default_height", 480))
        width = self._sample_int(sample, "width", self.config.get("inference", {}).get("default_width", 832))
        return self.model.dmd_latent_shape(batch_size, height, width)

    @staticmethod
    def _sample_int(sample, key, default):
        value = sample.get(key, default)
        if torch.is_tensor(value):
            value = value.detach().cpu().flatten()[0].item()
        if isinstance(value, (list, tuple)):
            value = value[0]
        return int(value)
