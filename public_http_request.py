"""Portable request-loop equivalent of the recorded public HTTP check."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import requests

p = argparse.ArgumentParser()
p.add_argument("--url", required=True)
p.add_argument("--images", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
images = sorted(args.images.glob("*.png"))
assert len(images) == 6
for label in ["warmup", "measure-1", "measure-2", "measure-3"]:
    data = {"prompt": "Create a short realistic video using the scenes and person shown in <Picture 1>, <Picture 2>, <Picture 3>, <Picture 4>, <Picture 5>, and <Picture 6>, with natural motion and ambient sound.",
            "width": "832", "height": "480", "fps": "24", "num_inference_steps": "1",
            "seed": "0", "flow_shift": "12.0",
            "extra_params": json.dumps({"task": "ref2va", "duration": 5.0, "audio_flow_shift": 3.0})}
    streams = [path.open("rb") for path in images]
    try:
        start = time.perf_counter()
        response = requests.post(args.url.rstrip("/")+"/v1/videos/sync", data=data,
            files=[("input_references", (path.name, stream, "image/png")) for path, stream in zip(images, streams)],
            timeout=7200)
        elapsed = (time.perf_counter()-start)*1000
    finally:
        for stream in streams:
            stream.close()
    response.raise_for_status()
    assert response.headers.get("content-type", "").startswith("video/mp4")
    output = args.output / (label+".mp4")
    output.write_bytes(response.content)
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"], check=True)
    result = {"label": label, "status": response.status_code, "wall_ms": elapsed,
              "bytes": len(response.content), "sha256": hashlib.sha256(response.content).hexdigest()}
    (args.output / (label+".json")).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
