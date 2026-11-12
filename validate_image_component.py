"""Public-image H3 VAE component validation; kept outside the PR diff."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
from PIL import Image
import torch
import torch_npu

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--boundary", choices=["encoder", "media"], default="encoder")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    source = Path(os.environ["OMNI_SOURCE_ROOT"])
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    images = [Image.open(path).convert("RGB") for path in sorted(args.images.glob("*.png"))]
    if not images:
        raise ValueError("No public images found")
    vae = MiniMaxH3VideoVAE(
        str(args.model), device=torch.device("npu:0"),
        load_device=torch.device("cpu"), trust_remote_code=True,
    ).eval()
    pipeline = media_input = None
    if args.boundary == "media":
        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
        from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput
        pipeline = object.__new__(MiniMaxH3Pipeline)
        torch.nn.Module.__init__(pipeline)
        pipeline.video_vae = vae
        pipeline.audio_vae = None
        pipeline.od_config = OmniDiffusionConfig(enable_layerwise_offload=True)
        pipeline._model_cpu_offload_modules = []
        media_input = MiniMaxH3EncoderMediaInput(
            task="ref2va", height=480, width=832, num_frames=124, latent_t=37, audio_t=207,
            images=tuple(torch.from_numpy(np.asarray(image).copy()) for image in images),
        )
    stager_calls = {"encoder":0, "decoder":0}
    for name in stager_calls:
        stager = getattr(vae, f"_{name}_stager")
        original_load = stager.load
        def counted_load(name=name, original_load=original_load):
            stager_calls[name] += 1
            return original_load()
        stager.load = counted_load
    hits = 0
    if (source / "vllm_omni/diffusion/models/minimax_h3/ops/vae/npu_image_conv.py").exists():
        from vllm_omni.diffusion.models.minimax_h3.ops.vae import npu_image_conv
        original = npu_image_conv._eligible
        def counted(module, tensor):
            nonlocal hits
            result = original(module, tensor)
            hits += int(result)
            return result
        npu_image_conv._eligible = counted
    results = []
    with torch.inference_mode():
        for trial in range(args.warmup + args.repeat):
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            before = hits
            loads_before = stager_calls.copy()
            start = time.perf_counter()
            if args.boundary == "media":
                conditioning = pipeline._encode_local_media(media_input)
                latents = [conditioning.visual_condition]
            else:
                vae.encoder_component.load_to_device()
                latents = [vae.encode_image(image) for image in images]
                vae.encoder_component.offload_to_cpu()
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            outputs = {}
            for i, latent in enumerate(latents):
                array = latent.detach().cpu().numpy()
                filename = f"trial-{trial}-image-{i}.npy"
                np.save(args.output / filename, array, allow_pickle=False)
                outputs[filename] = {"shape":list(array.shape), "dtype":str(array.dtype),
                    "sha256":hashlib.sha256(array.tobytes()).hexdigest(), "finite":bool(np.isfinite(array).all())}
            row = {"trial":trial, "kind":"warmup" if trial < args.warmup else "measure",
                "wall_ms":elapsed, "optimized_conv_calls":hits-before,
                "stager_loads":{name:stager_calls[name]-loads_before[name] for name in stager_calls},
                "peak_allocated_bytes":torch.npu.max_memory_allocated(), "outputs":outputs}
            results.append(row)
            (args.output / "progress.json").write_text(json.dumps(results, indent=2)+"\n")
            print(json.dumps({k:row[k] for k in ["trial","kind","wall_ms","optimized_conv_calls"]}), flush=True)
    measured = [row["wall_ms"] for row in results if row["kind"] == "measure"]
    result = {"boundary":"pipeline._encode_local_media including load/encode/offload" if args.boundary == "media" else "encoder load -> six image encodes -> encoder offload -> device synchronize",
        "source_commit":sha, "source_path":str(source), "public_model_revision":"42ed227ee7df40d41602854ae760620d6eb651fe",
        "torch":torch.__version__, "torch_npu":torch_npu.__version__, "device":torch.npu.get_device_name(0),
        "warmup_count":args.warmup, "measured_count":args.repeat, "results":results,
        "mean_ms":statistics.mean(measured), "stdev_ms":statistics.stdev(measured)}
    (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
