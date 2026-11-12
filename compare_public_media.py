"""Compare decoded public HTTP outputs, including each arm's self-repeat."""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np


def decode(path, kind):
    options, dtype = (
        (["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24"], np.uint8)
        if kind == "video"
        else (["-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le"], np.float32)
    )
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), *options, "-"])
    array = np.frombuffer(raw, dtype=dtype)
    return array, hashlib.sha256(raw).hexdigest()


def metrics(a, b, peak):
    if a.shape != b.shape:
        return {"shape_match": False, "base_elements": a.size, "candidate_elements": b.size}
    squared_error = squared_reference = 0.0
    changed = 0
    max_abs = 0.0
    finite = True
    for offset in range(0, a.size, 1 << 20):
        reference = a[offset : offset + (1 << 20)].astype(np.float64)
        candidate = b[offset : offset + (1 << 20)].astype(np.float64)
        diff = candidate - reference
        squared_error += float(np.sum(diff * diff))
        squared_reference += float(np.sum(reference * reference))
        changed += int(np.count_nonzero(diff))
        max_abs = max(max_abs, float(np.max(np.abs(diff))))
        finite &= bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    mse = squared_error / a.size
    return {
        "shape_match": True,
        "elements": int(a.size),
        "finite": finite,
        "exact": changed == 0,
        "max_abs": max_abs,
        "rmse": math.sqrt(mse),
        "relative_l2": math.sqrt(squared_error / squared_reference) if squared_reference else None,
        "psnr_db": 10 * math.log10(peak * peak / mse) if mse else None,
        "psnr_exact": mse == 0,
        "changed_fraction": changed / a.size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Precision root containing base and candidate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    media = {}
    pairs = {
        "base_self_repeat": ("base/warmup", "base/measure-1"),
        "candidate_self_repeat": ("candidate/warmup", "candidate/measure-1"),
        "paired_warmup": ("base/warmup", "candidate/warmup"),
        "paired_measure": ("base/measure-1", "candidate/measure-1"),
    }
    result = {
        "scope": "Decoded RGB24 video and float32 PCM, one measured pair; no quality pass inferred",
        "formula": "RMSE=sqrt(mean(diff^2)); relativeL2=norm(diff)/norm(reference); PSNR peak=255 video/1 PCM",
        "requests": {},
        "comparisons": {key: {} for key in pairs},
    }
    for kind in ("video", "audio"):
        for arm in ("base", "candidate"):
            for label in ("warmup", "measure-1"):
                key = f"{arm}/{label}"
                path = args.directory / f"{key}.mp4"
                array, sha = decode(path, kind)
                media[key] = array
                result["requests"].setdefault(key, {})[kind] = {
                    "decoded_sha256": sha,
                    "elements": int(array.size),
                    "dtype": str(array.dtype),
                }
        for label, (left, right) in pairs.items():
            result["comparisons"][label][kind] = metrics(media[left], media[right], 255 if kind == "video" else 1)
        media.clear()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
