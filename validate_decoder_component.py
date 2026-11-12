"""Real public H3 decoder A/B with an open seeded latent generator."""

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

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scheduled", action="store_true")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("hccl", timeout=timedelta(seconds=900))
    try:
        kwargs = {"enable_temporal_scheduling": True} if args.scheduled else {}
        vae = MiniMaxH3VideoVAE(args.model, device=device, load_device=torch.device("cpu"),
                               decode_only=True, trust_remote_code=True, **kwargs).eval()
        vae.set_parallel_size(dist.get_world_size(), process_group=dist.group.WORLD)
        selected = []
        if args.scheduled:
            from vllm_omni.diffusion.models.minimax_h3 import vae_window
            original = vae_window.select_temporal_window
            def counted(*call_args, **call_kwargs):
                value = original(*call_args, **call_kwargs)
                selected.append(value)
                return value
            vae_window.select_temporal_window = counted
        records = []
        args.output.mkdir(parents=True, exist_ok=True)
        for height, width in [(30, 52), (48, 84)]:
            latent = torch.randn((1, 24, 34, height, width), generator=torch.Generator().manual_seed(1101))
            input_sha = hashlib.sha256(latent.numpy().tobytes()).hexdigest()
            latent = latent.to(device)
            for trial in range(args.warmup + args.repeat):
                torch.npu.synchronize()
                dist.barrier()
                torch.npu.reset_peak_memory_stats()
                before = len(selected)
                start = time.perf_counter()
                vae.decoder_component.load_to_device()
                with torch.inference_mode(), torch.autocast("npu", dtype=torch.float16):
                    decoded = vae.decode_latent(latent)
                vae.decoder_component.offload_to_cpu()
                torch.npu.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                array = decoded.cpu().numpy()
                row = {"rank": rank, "height": height*16, "width": width*16,
                       "trial": trial, "kind": "warmup" if trial < args.warmup else "measure",
                       "wall_ms": elapsed, "input_sha256": input_sha,
                       "selected_windows": selected[before:], "shape": list(array.shape),
                       "dtype": str(array.dtype), "finite": bool(np.isfinite(array).all()),
                       "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                       "peak_allocated_bytes": torch.npu.max_memory_allocated()}
                if rank == 0:
                    np.save(args.output / f"{width*16}x{height*16}-trial{trial}.npy", array, allow_pickle=False)
                records.append(row)
                (args.output / f"rank{rank}-progress.json").write_text(json.dumps(records, indent=2)+"\n")
                print(json.dumps({k: row[k] for k in ["rank", "width", "height", "trial", "wall_ms", "selected_windows"]}), flush=True)
                del decoded, array
            del latent
        source = Path(os.environ["OMNI_SOURCE_ROOT"])
        report = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
                  "source_path": str(source), "world_size": dist.get_world_size(), "rank": rank,
                  "torch": torch.__version__, "torch_npu": torch_npu.__version__, "scheduled": args.scheduled,
                  "input": "open seeded Gaussian latent, seed1101,N1,C24,T34; public VAE weights",
                  "scope": "decoder load, normalized full decode, offload, synchronization; no DiT/audio/mux",
                  "warmup": args.warmup, "repeat": args.repeat, "records": records}
        (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2)+"\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
