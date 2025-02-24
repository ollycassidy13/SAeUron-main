from dataclasses import dataclass
import torch
from simple_parsing import Serializable, list_field

@dataclass
class SaeConfig(Serializable):
    expansion_factor: int = 32
    normalize_decoder: bool = True
    num_latents: int = 0
    k: int = 32
    batch_topk: bool = False
    sample_topk: bool = False
    input_unit_norm: bool = False
    multi_topk: bool = False

@dataclass
class TrainConfig(Serializable):
    sae: SaeConfig
    dataset_path: list[str] = list_field()
    effective_batch_size: int = 4096
    num_workers: int = 1
    persistent_workers: bool = True
    prefetch_factor: int = 2
    grad_acc_steps: int = 1
    micro_acc_steps: int = 1
    lr: float | None = None
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 1000
    auxk_alpha: float = 0.0
    dead_feature_threshold: int = 10_000_000
    feature_sampling_window: int = 100
    hookpoints: list[str] = list_field()
    distribute_modules: bool = False
    save_every: int = 5000
    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 1
    wandb_project: str = "sae_styles-hook0"

    def __post_init__(self):
        if self.run_name is None:
            variant = "patch_topk"
            if self.sae.batch_topk:
                variant = "batch_topk"
            elif self.sae.sample_topk:
                variant = "sample_topk"
            self.run_name = (
                f"{variant}_expansion_factor{self.sae.expansion_factor}"
                f"_k{self.sae.k}_multi_topk{self.sae.multi_topk}"
                f"_auxk_alpha{self.auxk_alpha}"
            )

@dataclass
class CacheActivationsRunnerConfig:
    hook_names: list[str] | None = None
    new_cached_activations_path: str | None = None
    dataset_name: str = "guangyil/laion-coco-aesthetic"
    split: str = "train"
    column: str = "caption"
    device: torch.device | str = "cuda"
    model_name: str = "valhalla/sd-wikiart-v2"
    dtype: torch.dtype = torch.float16
    num_inference_steps: int = 50
    seed: int = 42
    batch_size: int = 10
    num_workers: int = 8
    output_or_diff: str = "output"
    max_num_examples: int | None = None
    cache_every_n_timesteps: int = 1
    guidance_scale: float = 9.0
    class_start: int = 0
    class_end: int = 20

    hf_repo_id: str | None = None
    hf_num_shards: int | None = None
    hf_revision: str = "main"
    hf_is_private_repo: bool = False

    # NEW: parameters for incremental caching/training
    num_chunks: int = 5
    rolling_buffer_keep_chunks: int = 1

    def __post_init__(self):
        if self.new_cached_activations_path is None:
            self.new_cached_activations_path = (
                f"activations/{self.dataset_name.split('/')[-1]}/"
                f"{self.model_name.split('/')[-1]}/{self.output_or_diff}/"
            )
        if isinstance(self.hook_names, str):
            self.hook_names = [self.hook_names]
