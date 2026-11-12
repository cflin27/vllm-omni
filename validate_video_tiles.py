"""Public reference-video encoder comparison on a real native tile group."""

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

import vllm_omni
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--public-frames", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tasks", type=int)
    p.add_argument("--lengths", type=int, nargs="+", default=[39, 156, 312])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("hccl", timeout=timedelta(seconds=900))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        source = Path(os.environ["OMNI_SOURCE_ROOT"])
        assert Path(vllm_omni.__file__).is_relative_to(source)
        kwargs = {} if args.max_tasks is None else {"encode_tile_balance_max_tasks": args.max_tasks}
        vae = MiniMaxH3VideoVAE(args.model, device=device, load_device=torch.device("cpu"),
                               encode_only=True, trust_remote_code=True, **kwargs).eval()
        vae.set_parallel_size(world, process_group=dist.group.WORLD)
        tile_calls = 0
        original = vae.model.encode
        def counted(*call_args, **call_kwargs):
            nonlocal tile_calls
            tile_calls += 1
            return original(*call_args, **call_kwargs)
        vae.model.encode = counted
        public = np.load(args.public_frames, allow_pickle=False)
        records = []
        args.output.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for length in args.lengths:
                # Loop the same public clip to exercise long/multiple windows.
                frames = np.tile(public, (int(np.ceil(length / len(public))), 1, 1, 1))[:length].copy()
                input_sha = hashlib.sha256(frames.tobytes()).hexdigest()
                for trial in range(args.warmup + args.repeat):
                    torch.npu.synchronize()
                    dist.barrier()
                    torch.npu.reset_peak_memory_stats()
                    before = tile_calls
                    start = time.perf_counter()
                    vae.encoder_component.load_to_device()
                    latent, shape = vae.encode_video(frames)
                    vae.encoder_component.offload_to_cpu()
                    torch.npu.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    array = latent.numpy()
                    row = {"rank": rank, "frames": length, "input_sha256": input_sha,
                           "trial": trial, "kind": "warmup" if trial < args.warmup else "measure",
                           "wall_ms": elapsed, "tile_forward_calls": tile_calls-before,
                           "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                           "latent_shape": list(shape), "array_shape": list(array.shape),
                           "finite": bool(np.isfinite(array).all()),
                           "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
                    if rank == 0:
                        np.save(args.output / f"frames{length}-trial{trial}.npy", array, allow_pickle=False)
                    records.append(row)
                    (args.output / f"rank{rank}-progress.json").write_text(json.dumps(records, indent=2)+"\n")
                    print(json.dumps({k:row[k] for k in ["rank", "frames", "trial", "wall_ms", "tile_forward_calls"]}), flush=True)
                del frames
        report = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
                  "source_path": str(source), "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  "torch": torch.__version__, "torch_npu": torch_npu.__version__, "world_size": world,
                  "rank": rank, "device": torch.npu.get_device_name(local_rank), "max_tasks": args.max_tasks,
                  "warmup": args.warmup, "repeat": args.repeat, "lengths": args.lengths,
                  "scope": "encoder load, public frame prepare/encode, CPU latent processing, offload, sync",
                  "records": records}
        (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2)+"\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
