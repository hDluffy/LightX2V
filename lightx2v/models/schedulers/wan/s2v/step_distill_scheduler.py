import torch

from lightx2v.models.schedulers.wan.s2v.s2v_scheduler import WanS2VScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class WanS2VStepDistillScheduler(WanS2VScheduler):
    """Fixed-step Flow Matching scheduler for Wan2.2-S2V step-distilled LoRA inference."""

    def __init__(self, config):
        super().__init__(config)
        self.denoising_step_list = config["denoising_step_list"]
        self.infer_steps = len(self.denoising_step_list)
        self.sigma_max = 1.0
        self.sigma_min = 0.0

    def prepare_clip(self, seed, latent_shape, dtype):
        self.generator = torch.Generator(device=AI_DEVICE).manual_seed(seed)
        self.latents = torch.randn(
            *latent_shape,
            dtype=dtype,
            device=AI_DEVICE,
            generator=self.generator,
        )
        self.set_denoising_timesteps(device=AI_DEVICE)
        self.step_index = 0

    def set_denoising_timesteps(self, device=None):
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, self.num_train_timesteps + 1, dtype=torch.float32)[:-1]
        sigmas = self.sample_shift * sigmas / (1 + (self.sample_shift - 1) * sigmas)
        timesteps = sigmas * self.num_train_timesteps

        indices = [self.num_train_timesteps - int(step) for step in self.denoising_step_list]
        self.sigmas = sigmas[indices].to("cpu")
        self.timesteps = timesteps[indices].to(device=device, dtype=torch.float32)

    def step_pre(self, step_index):
        self.step_index = step_index
        self.timestep_input = torch.stack([self.timesteps[step_index]])

    def step_post(self):
        if getattr(self, "noise_pred", None) is None:
            return
        flow_pred = self.noise_pred.to(torch.float32)
        sigma = self.sigmas[self.step_index].item()
        latents = self.latents.to(torch.float32) - sigma * flow_pred
        if self.step_index < self.infer_steps - 1:
            sigma_next = self.sigmas[self.step_index + 1].item()
            latents = latents + sigma_next * flow_pred
        self.latents = latents.to(self.latents.dtype)
