import io
import json
import math
import os
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from diffusers.utils.import_utils import is_xformers_available

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Array2D, Dataset, Features, Value, concatenate_datasets
from datasets.fingerprint import generate_fingerprint
from huggingface_hub import HfApi
from tqdm import tqdm

from SAE.config import CacheActivationsRunnerConfig
from UnlearnCanvas_resources.const import class_available, theme_available

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

TORCH_STRING_DTYPE_MAP = {torch.float16: "float16", torch.float32: "float32"}


class CacheActivationsRunner:
    def __init__(self, cfg: CacheActivationsRunnerConfig):
        self.cfg = cfg
        self.accelerator = Accelerator()

        if self.cfg.hook_names:
            from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline

            self.pipe = HookedStableDiffusionPipeline.from_pretrained(
                self.cfg.model_name, torch_dtype=self.cfg.dtype, safety_checker=None
            )
            if is_xformers_available():
                print("Enabling xFormers memory efficient attention")
                self.pipe.unet.enable_xformers_memory_efficient_attention()
            self.pipe.to(self.accelerator.device)
            self.pipe.vae.to("cpu")
            self.pipe.set_progress_bar_config(disable=True)

            self.scheduler = self.pipe.scheduler
            self.scheduler.set_timesteps(self.cfg.num_inference_steps, device="cpu")
            self.scheduler_timesteps = self.scheduler.timesteps

            self.features_dict = {hook: None for hook in self.cfg.hook_names}

    def _get_chunk_prompts(self, chunk_idx: int, lines_per_chunk: int) -> list[str]:
        """
        Reads the lines [chunk_idx*lines_per_chunk : (chunk_idx+1)*lines_per_chunk]
        from each noun file, appends styles, and shuffles them within the chunk.
        """
        chunk_prompts = []
        for noun in class_available[self.cfg.class_start : self.cfg.class_end]:
            path = os.path.join(
                "UnlearnCanvas_resources",
                "anchor_prompts",
                "finetune_prompts",
                f"sd_prompt_{noun}.txt",
            )
            with open(path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines()]
            start = chunk_idx * lines_per_chunk
            end = start + lines_per_chunk
            chunk_lines = lines[start:end]
            for line in chunk_lines:
                if line.endswith("."):
                    line = line[:-1]
                for theme in theme_available:
                    chunk_prompts.append(f"{line} in {theme.replace('_', ' ')} style.")
                chunk_prompts.append(line + ".")
        random.shuffle(chunk_prompts)
        return chunk_prompts

    @staticmethod
    def get_batches(items: list[str], batch_size: int):
        num_batches = (len(items) + batch_size - 1) // batch_size
        batches = []
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(items))
            batches.append(items[start_index:end_index])
        return batches

    @staticmethod
    def _consolidate_shards(source_dir: Path, output_dir: Path, copy_files: bool = True) -> Dataset:
        first_shard_dir_name = "shard_00000"
        assert source_dir.exists() and source_dir.is_dir()
        assert (
            output_dir.exists()
            and output_dir.is_dir()
            and not any(p for p in output_dir.iterdir() if not p.name == ".tmp_shards")
        )
        if not (source_dir / first_shard_dir_name).exists():
            raise Exception(f"No shards in {source_dir} exist!")
        transfer_fn = shutil.copy2 if copy_files else shutil.move
        transfer_fn(
            source_dir / first_shard_dir_name / "dataset_info.json",
            output_dir / "dataset_info.json",
        )
        arrow_files = []
        file_count = 0
        for shard_dir in sorted(source_dir.iterdir()):
            if not shard_dir.name.startswith("shard_"):
                continue
            state = json.loads((shard_dir / "state.json").read_text())
            for data_file in state["_data_files"]:
                src = shard_dir / data_file["filename"]
                new_name = f"data-{file_count:05d}-of-{len(list(source_dir.iterdir())):05d}.arrow"
                dst = output_dir / new_name
                transfer_fn(src, dst)
                arrow_files.append({"filename": new_name})
                file_count += 1
        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }
        with open(output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)
        ds = Dataset.load_from_disk(str(output_dir))
        fingerprint = generate_fingerprint(ds)
        del ds
        with open(output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()
        if not copy_files:
            shutil.rmtree(source_dir)
        return Dataset.load_from_disk(output_dir)

    @torch.no_grad()
    def _create_shard(self, buffer: torch.Tensor, hook_name: str) -> Dataset:
        batch_size, n_steps, d_sample_size, d_in = buffer.shape
        buffer = buffer[:, :: self.cfg.cache_every_n_timesteps, :, :]
        activations = buffer.reshape(-1, d_sample_size, d_in)
        timesteps = self.scheduler_timesteps[:: self.cfg.cache_every_n_timesteps].repeat(batch_size)
        shard = Dataset.from_dict(
            {"activations": activations, "timestep": timesteps},
            features=self.features_dict[hook_name],
        )
        return shard

    def create_dataset_feature(self, hook_name: str, d_in: int, d_out: int):
        self.features_dict[hook_name] = Features({
            "activations": Array2D(shape=(d_in, d_out), dtype=TORCH_STRING_DTYPE_MAP[self.cfg.dtype]),
            "timestep": Value(dtype="uint16"),
        })

    @torch.no_grad()
    def run(self) -> dict[str, Dataset]:
        # Assume each noun file has exactly 80 lines; with num_chunks=5, each chunk uses 16 lines.
        lines_per_file = 80
        lines_per_chunk = lines_per_file // self.cfg.num_chunks

        # Setup final directories for each hook.
        assert self.cfg.new_cached_activations_path is not None
        final_cached_activation_paths = {
            n: Path(os.path.join(self.cfg.new_cached_activations_path, n))
            for n in self.cfg.hook_names
        }
        if self.accelerator.is_main_process:
            for path in final_cached_activation_paths.values():
                path.mkdir(exist_ok=True, parents=True)
                if any(path.iterdir()):
                    raise Exception(
                        f"Activations directory ({path}) is not empty. Please delete it or specify a different path."
                    )
            # Create rolling buffer directories.
            rolling_buffer_paths = {}
            for hook, path in final_cached_activation_paths.items():
                rb_path = path / "rolling_buffer"
                rb_path.mkdir(exist_ok=True, parents=True)
                rolling_buffer_paths[hook] = rb_path
        else:
            rolling_buffer_paths = {}

        self.accelerator.wait_for_everyone()

        # Process each chunk (chunk 0 to num_chunks-1)
        for chunk_idx in range(self.cfg.num_chunks):
            if self.accelerator.is_main_process:
                print(f"\n=== Processing chunk {chunk_idx+1}/{self.cfg.num_chunks} ===")

            # 1) Get prompts for this chunk (each noun file contributes its designated lines)
            chunk_prompts = self._get_chunk_prompts(chunk_idx, lines_per_chunk)

            # 2) Create batches for inference
            batches = self.get_batches(chunk_prompts, self.cfg.batch_size)

            # 3) Create temporary subdirectories for this chunk's shards
            tmp_chunk_paths = {
                n: final_cached_activation_paths[n] / f"chunk_{chunk_idx:05d}" / ".tmp_shards"
                for n in self.cfg.hook_names
            }
            if self.accelerator.is_main_process:
                for p in tmp_chunk_paths.values():
                    p.parent.mkdir(exist_ok=True, parents=True)
                    p.mkdir(exist_ok=True)

            # 4) For each batch, run inference and save shards
            for i, batch_prompts in tqdm(
                enumerate(batches),
                desc=f"Caching chunk {chunk_idx+1}",
                total=len(batches),
                disable=not self.accelerator.is_main_process
            ):
                with self.accelerator.split_between_processes(batch_prompts) as local_prompts:
                    _, acts_cache = self.pipe.run_with_cache(
                        prompt=local_prompts,
                        output_type="latent",
                        num_inference_steps=self.cfg.num_inference_steps,
                        save_input=True if self.cfg.output_or_diff == "diff" else False,
                        save_output=True,
                        positions_to_cache=self.cfg.hook_names,
                        guidance_scale=self.cfg.guidance_scale,
                    )
                self.accelerator.wait_for_everyone()

                gathered_buffer = {}
                for hook_name in self.cfg.hook_names:
                    if self.cfg.output_or_diff == "diff":
                        gathered_buffer[hook_name] = (
                            acts_cache["output"][hook_name] - acts_cache["input"][hook_name]
                        )
                    else:
                        gathered_buffer[hook_name] = acts_cache["output"][hook_name]
                gathered_buffer = gather_object([gathered_buffer])
                if self.accelerator.is_main_process:
                    for hook_name in self.cfg.hook_names:
                        gathered_acts = torch.cat(
                            [g[hook_name] for g in gathered_buffer],
                            dim=0
                        )
                        if self.features_dict[hook_name] is None:
                            self.create_dataset_feature(
                                hook_name,
                                gathered_acts.shape[-2],
                                gathered_acts.shape[-1],
                            )
                        shard = self._create_shard(gathered_acts, hook_name)
                        shard.save_to_disk(
                            f"{tmp_chunk_paths[hook_name]}/shard_{i:05d}",
                            num_shards=1,
                        )
                        del gathered_acts, shard
                    del gathered_buffer

            # 5) Consolidate the chunk's shards into a single dataset
            if self.accelerator.is_main_process:
                for hook_name, base_path in final_cached_activation_paths.items():
                    chunk_dir = base_path / f"chunk_{chunk_idx:05d}"
                    consolidated = self._consolidate_shards(
                        chunk_dir / ".tmp_shards", chunk_dir, copy_files=False
                    )
                    print(f"Consolidated dataset for {hook_name}, chunk {chunk_idx+1}")
                    shutil.rmtree(chunk_dir / ".tmp_shards")

            # 6) Update the rolling buffer by merging the current chunk with previous data.
            if self.accelerator.is_main_process:
                from datasets import load_from_disk
                for hook_name in self.cfg.hook_names:
                    chunk_dir = final_cached_activation_paths[hook_name] / f"chunk_{chunk_idx:05d}"
                    current_dataset = load_from_disk(str(chunk_dir))
                    rb_path = rolling_buffer_paths[hook_name]
                    if any(rb_path.iterdir()):
                        rb_dataset = load_from_disk(str(rb_path))
                        combined_dataset = concatenate_datasets([rb_dataset, current_dataset])
                    else:
                        combined_dataset = current_dataset
                    # Optionally, shuffle the combined dataset.
                    combined_dataset = combined_dataset.shuffle(seed=42)
                    combined_dataset.save_to_disk(str(rb_path))

            # 7) Train the SAE on the rolling buffer (which now contains both old and current chunk activations)
            if self.accelerator.is_main_process:
                print(f"=== Training SAE on rolling buffer after chunk {chunk_idx+1} ===")
                from SAE.config import TrainConfig, SaeConfig
                from SAE.trainer import SaeTrainer

                dataset_dict = {}
                for hook_name in self.cfg.hook_names:
                    ds = load_from_disk(str(rolling_buffer_paths[hook_name]))
                    ds.set_format(
                        type="torch",
                        columns=["activations", "timestep"],
                        dtype=torch.float32,
                    )
                    dataset_dict[hook_name] = ds

                train_cfg = TrainConfig(
                    sae=SaeConfig(),
                    dataset_path=[str(rolling_buffer_paths[h]) for h in self.cfg.hook_names],
                    effective_batch_size=4096,
                    num_workers=1,
                    grad_acc_steps=1,
                    micro_acc_steps=1,
                    lr=2e-4,
                    hookpoints=self.cfg.hook_names,
                )
                print("Training configuration:", train_cfg)

                trainer = SaeTrainer(train_cfg, dataset_dict)
                trainer.fit()

                # 7b) After training on this chunk, push the SAE to Hugging Face Hub if configured.
                if self.cfg.hf_repo_id is not None:
                    print(f"Pushing SAE to Hugging Face Hub after chunk {chunk_idx+1}...")
                    for hook_name, sae in trainer.saes.items():
                        sae.push_to_hub(
                            repo_id=f"{self.cfg.hf_repo_id}_{hook_name}",
                            revision=self.cfg.hf_revision,
                            private=self.cfg.hf_is_private_repo,
                        )

                # 8) Delete the current chunk's cached data (keeping the rolling buffer intact).
                for hook_name, base_path in final_cached_activation_paths.items():
                    chunk_dir = f"{base_path}/chunk_{chunk_idx:05d}"
                    if chunk_dir.exists():
                        shutil.rmtree(chunk_dir)
                        print(f"Deleted chunk data for {hook_name}, chunk {chunk_idx+1}")

        # End of chunk loop.
        from datasets import load_from_disk
        final_datasets = {}
        if self.accelerator.is_main_process:
            for hook_name, rb_path in rolling_buffer_paths.items():
                final_datasets[hook_name] = load_from_disk(str(rb_path))
        return final_datasets


def load_and_push_to_hub() -> None:
    from SAE.config import CacheActivationsRunnerConfig
    cfg = CacheActivationsRunnerConfig()
    runner = CacheActivationsRunner(cfg)
    runner.run()
