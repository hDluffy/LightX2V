from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from lightx2v_train.runtime.distributed import get_world_size
from lightx2v_train.utils.registry import DATA_REGISTER


class WanS2VCachedDataset(torch.utils.data.Dataset):
    """Dataset for precomputed Wan2.2-S2V training conditions.

    Each .pt file should contain context/context_null, ref_latents,
    motion_latents, cond_latents and audio_input/audio_emb. A latent target is
    optional for DMD, but latent_shape must be available or inferable.
    """

    def __init__(self, cache_paths, dataset_repeat=1, max_samples=None):
        if isinstance(cache_paths, (str, Path)):
            cache_paths = [cache_paths]
        self.cache_paths = self._collect_cache_paths(cache_paths)
        if max_samples is not None:
            self.cache_paths = self.cache_paths[:max_samples]
        self.dataset_repeat = int(dataset_repeat)
        if not self.cache_paths:
            raise RuntimeError(f"No .pt cache files found from cache_paths={cache_paths}")

    def _collect_cache_paths(self, cache_paths):
        result = []
        for cache_path in cache_paths:
            path = Path(cache_path)
            if path.is_dir():
                result.extend(sorted(str(item) for item in path.rglob("*.pt")))
            elif path.suffix == ".pt":
                result.append(str(path))
            elif path.suffix in {".txt", ".list"}:
                with path.open("r", encoding="utf-8") as handle:
                    result.extend(line.strip() for line in handle if line.strip())
        return result

    def __getitem__(self, index):
        path = self.cache_paths[index % len(self.cache_paths)]
        item = torch.load(path, map_location="cpu", weights_only=False)
        item = self._flatten_known_groups(item)
        item = self._normalize_aliases(item)
        item = self._squeeze_cached_batch_dims(item)
        item["cache_path"] = path
        item.setdefault("prompt", "")
        if "latent_shape" not in item:
            item["latent_shape"] = torch.tensor(self._infer_latent_shape(item), dtype=torch.long)
        elif not torch.is_tensor(item["latent_shape"]):
            item["latent_shape"] = torch.tensor(item["latent_shape"], dtype=torch.long)
        if "motion_frames" not in item:
            motion_frames = int(item.get("motion_frame_count", 73))
            item["motion_frames"] = torch.tensor([motion_frames, (motion_frames + 3) // 4], dtype=torch.long)
        elif not torch.is_tensor(item["motion_frames"]):
            item["motion_frames"] = torch.tensor(item["motion_frames"], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.cache_paths) * self.dataset_repeat

    def _flatten_known_groups(self, item):
        result = dict(item)
        text = result.pop("text_encoder_output", None)
        if isinstance(text, dict):
            result.setdefault("context", text.get("context"))
            result.setdefault("context_null", text.get("context_null"))
        s2v = result.pop("s2v", None)
        if isinstance(s2v, dict):
            for key, value in s2v.items():
                result.setdefault(key, value)
        return result

    @staticmethod
    def _normalize_aliases(item):
        if "context" not in item and "prompt_embed" in item:
            item["context"] = item["prompt_embed"]
        if "context_null" not in item and "negative_prompt_embed" in item:
            item["context_null"] = item["negative_prompt_embed"]
        if "audio_input" not in item and "audio_emb" in item:
            item["audio_input"] = item["audio_emb"]
        return item

    @staticmethod
    def _squeeze_cached_batch_dims(item):
        for key in (
            "context",
            "context_null",
            "prompt_embed",
            "negative_prompt_embed",
            "ref_latents",
            "motion_latents",
            "cond_latents",
            "audio_input",
            "audio_emb",
            "latent",
            "target_latent",
        ):
            value = item.get(key)
            if torch.is_tensor(value) and value.ndim >= 3 and value.shape[0] == 1:
                item[key] = value[0]
        return item

    @staticmethod
    def _infer_latent_shape(item):
        latent = item.get("latent", item.get("target_latent"))
        if torch.is_tensor(latent):
            return tuple(latent.shape if latent.ndim == 5 else (1, *latent.shape))
        cond_latents = item.get("cond_latents")
        if torch.is_tensor(cond_latents):
            if cond_latents.ndim == 4:
                return tuple(cond_latents.shape)
            if cond_latents.ndim == 5:
                return tuple(cond_latents.shape[1:] if cond_latents.shape[0] == 1 else cond_latents.shape)
        raise RuntimeError("S2V cache must contain latent_shape, latent/target_latent, or cond_latents.")


def _build_dataloader(dataset, data_config, train_or_val):
    world_size = get_world_size()
    sampler = None
    shuffle = data_config.get("shuffle", train_or_val == "train")
    drop_last = data_config.get("drop_last", False)
    if train_or_val == "train" and world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=data_config.get("batch_size", 1),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=data_config.get("num_workers", 8),
        pin_memory=data_config.get("pin_memory", True),
        drop_last=drop_last if sampler is None else False,
    )


@DATA_REGISTER("wan_s2v_cached_dataset")
def build_wan_s2v_cached_dataset(data_config, train_or_val="train"):
    cache_paths = data_config.get("cache_path", data_config.get("data_path"))
    dataset = WanS2VCachedDataset(
        cache_paths=cache_paths,
        dataset_repeat=data_config.get("dataset_repeat", 1),
        max_samples=data_config.get("max_samples"),
    )
    return _build_dataloader(dataset, data_config, train_or_val)
