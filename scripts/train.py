import os
import sys
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from simple_parsing import parse

from SAE.config import TrainConfig
from SAE.trainer import SaeTrainer

@dataclass
class RunConfig(TrainConfig):
    mixed_precision: str = "no"
    max_examples: int | None = None
    seed: int = 42
    device: str = "cuda"
    num_epochs: int = 1

def load_datasets_from_dirs(base_dirs, hookpoint, dtype=torch.float32):
    datasets = []
    print(f"Concatenating datasets from {base_dirs}")
    for base_dir in base_dirs:
        dataset = Dataset.load_from_disk(os.path.join(base_dir, hookpoint), keep_in_memory=False)
        dataset.set_format(type="torch", columns=["activations", "timestep"], dtype=dtype)
        datasets.append(dataset)
    return concatenate_datasets(datasets)

def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0
    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")
        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")
    args = parse(RunConfig)
    args.run_name = args.run_name + f"_{args.dataset_path[0].split('/')[-2]}"
    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    args.dtype = dtype
    print(f"Training in {dtype=}")
    dataset_dict = {}
    if not ddp or rank == 0:
        for hookpoint in args.hookpoints:
            if len(args.dataset_path) > 1:
                dataset = load_datasets_from_dirs(args.dataset_path, hookpoint, dtype)
            else:
                dataset = Dataset.load_from_disk(
                    os.path.join(args.dataset_path[0], hookpoint), keep_in_memory=False
                )
            dataset.set_format(type="torch", columns=["activations", "timestep"], dtype=dtype)
            dataset = dataset.shuffle(args.seed)
            if args.max_examples:
                dataset = dataset.select(range(args.max_examples))
            dataset_dict[hookpoint] = dataset
            print(f"Loaded dataset for {hookpoint}")
    if ddp:
        dist.barrier()
        if rank != 0:
            for hookpoint in args.hookpoints:
                dataset = Dataset.load_from_disk(
                    os.path.join(args.dataset_path, hookpoint), keep_in_memory=False
                )
                dataset.set_format(type="torch", columns=["activations", "timestep"], dtype=dtype)
                dataset = dataset.shuffle(args.seed)
                dataset = dataset.shard(dist.get_world_size(), rank)
                dataset_dict[hookpoint] = dataset
                print(f"Loaded dataset for {hookpoint}")
    with nullcontext() if rank == 0 else redirect_stdout(None):
        trainer = SaeTrainer(args, dataset_dict)
        trainer.fit()

if __name__ == "__main__":
    run()
