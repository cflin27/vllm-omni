# Public MiniMax H3 PR validation

Inputs are six images extracted from the official vLLM-Omni Ref2VA input video. Model revision: `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`. No private cases or weights are distributed.

Hardware: Ascend 910B4; one physical NPU per component arm. Software: Python 3.12.13, torch 2.10.0+cpu, torch-npu 2.10.0.post7, vLLM 0.33.0+empty, vLLM-Ascend e617b709, CANN 9.1.0, driver 25.5.0; FFmpeg 4.4.2. `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`; FP32 VAE, default HF32 policy, deterministic algorithms OFF in component timing. The attached Ascend QSA import patch was applied identically to both arms. The exact remaining package versions are in `runtime-package-versions.json`. Use an isolated NPU runtime with matching dependencies; do not modify an existing service environment.

## Inputs

```bash
hf download MiniMaxAI/MiniMax-H3 --revision 42ed227ee7df40d41602854ae760620d6eb651fe --include 'Ref2VA/video_vae/**' --local-dir "$PUBLIC_MODEL"
curl -fL 'https://cdn.hailuoai.com/prod/hailuo_demo/testsets/h3_promo_eval_ref2va/gallery/sr_v2p26_trio_seed42_20260724/inputs/297573323635_00_%E8%A7%86%E9%A2%911_YnyRbxEwio_video_20260525_163755_1927e9d3.mp4' -o public-input.mp4
printf '%s\n' 'd6edc20ac82aa980c9cbd0b05b371f60d12dc8d018492e571b42de0086819c41  public-input.mp4' | sha256sum -c -
mkdir public-images
ffmpeg -nostdin -v error -i public-input.mp4 -vf 'select=eq(n\,0)+eq(n\,24)+eq(n\,48)+eq(n\,72)+eq(n\,96)+eq(n\,120),scale=832:480:flags=lanczos' -vsync vfr public-images/public-ref-%02d.png
```

These resized public frames are component performance inputs, not the upstream official quality reference. `public-inputs.json` records their hashes. The VAE weight file was independently verified against the download LFS SHA256 `5f0c2e161d895a9fee7645ca32d4a7e3a22b90cacfcbeba62ec999cdbbefe0d3` (10,415,548,320 bytes).

## Component commands

Use clean source checkouts and the attached harness, with the same Python/dependencies for both arms. Baseline: `b7bbfbdb4f2b3c070d9632043ea6cd2b1c601270`; PR9467: `a801490a57a50f0d047b2dcf29581ee0d50811f2`; PR9494: `343bec72066afed3cf9ce9d69d8202ae30e69841`. Candidate source and exact changed-file hashes are in the attached result/hash files. In each clean checkout, run `python setup.py --version` with the isolated Python before importing source; this generates its source-local `_version.py`. Fetch `refs/pull/9467/head` and `refs/pull/9494/head` from the public Omni repository and verify these commits, rather than using a moving branch silently.

```bash
PUBLIC_RUN_ROOT="$(pwd)"
ASCEND_RT_VISIBLE_DEVICES=0 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True OMNI_SOURCE_ROOT="$BASE_CHECKOUT" PYTHONPATH="$BASE_CHECKOUT" python "$PUBLIC_RUN_ROOT/pr_python.py" "$PUBLIC_RUN_ROOT/validate_image_component.py" --boundary encoder --model "$PUBLIC_MODEL/Ref2VA/video_vae" --images "$PUBLIC_RUN_ROOT/public-images" --output "$NEW_OUTPUT/base" --warmup 2 --repeat 3
ASCEND_RT_VISIBLE_DEVICES=0 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True OMNI_SOURCE_ROOT="$PR9467_CHECKOUT" PYTHONPATH="$PR9467_CHECKOUT" python "$PUBLIC_RUN_ROOT/pr_python.py" "$PUBLIC_RUN_ROOT/validate_image_component.py" --boundary encoder --model "$PUBLIC_MODEL/Ref2VA/video_vae" --images "$PUBLIC_RUN_ROOT/public-images" --output "$NEW_OUTPUT/candidate" --warmup 2 --repeat 3
# PR9494 instead uses --boundary media to invoke the actual _encode_local_media method.
python compare_component.py "$NEW_OUTPUT/base" "$NEW_OUTPUT/candidate" > "$NEW_OUTPUT/comparison.json"
```

For an OCI image with editable installs pointing outside these checkouts, use the attached `pr_python.py` entry to exclude editable finders. Import paths must resolve to the selected public checkout. The component timer includes encoder load, six image encodes, offload and final device synchronization. Initialization, hashing and array export are outside the timer. Raw per-call times, shapes, hashes, actual optimized Conv3D counts and stager-routing counts are retained. Compare corresponding `.npy` files with `np.array_equal`; warmups are excluded from the means.

The component results do not establish HTTP, 50-step quality, throughput or long-run stability. End-to-end deterministic 1-NFE accuracy is reported separately on the PR. No device profile or formal E2E speedup is inferred from these component values.

## Recorded runs and limits

| PR / run | Base mean ms | Candidate mean ms | Latency reduction | Exact arrays, including warmups |
| --- | --- | --- | --- | --- |
| 9467 / initial, base first | 2565.949 | 902.484 | 64.83% | 30/30 |
| 9467 / frozen harness repeat, candidate first | 2579.333 | 1669.207 | 35.29% | 30/30 |
| 9494 / production media boundary, base first | 4144.515 | 2550.478 | 38.46% | 5/5 |

Each arm runs in a separate process, with two warmups and three unprofiled measurements. Both PR9467 runs observe 2448 optimized Conv3D calls/request. The gain magnitude differs substantially between runs; the cause is unresolved. These runs are retained separately, without pooling or a general 64.83% production claim. The initial run did not retain an exact harness snapshot; the attached snapshot was used for the repeat and for PR9494. Per-call timing and array hashes are attached. The exact source/test hashes match both candidate commits.

PR9494 stager loads are encoder/decoder = 1/1 before and 1/0 after, every request. Peak allocated bytes are 10,778,363,392 / 1,084,004,352 in this isolated component. This is component allocation, not a full-pipeline peak or measured DMA volume.

PR9467 also completed one warmup and three deterministic HTTP requests per arm with the same six images: all eight MP4s are byte-identical, finite media decoded with FFmpeg, SHA256 `ffd4c583a31d456ad4a39e7f3705ebaf97eb21f4bbc26dae6132bb5b87572a64`. These are 1-NFE correctness observations. Service termination emitted TBE finalizer errors after the final successful requests; clean native shutdown and long-term stability are not established. PR9494 did not run this HTTP comparison.

For that correctness check, download the complete public model at the same revision and launch each source checkout separately with deterministic algorithms enabled in the server/spawn processes:

```bash
python "$PUBLIC_RUN_ROOT/pr_python.py" -m vllm_omni.entrypoints.cli.main serve "$PUBLIC_MODEL" --omni --host 127.0.0.1 --port 18680 --trust-remote-code --task-type ref2va --num-gpus 8 --model-class-name MiniMaxH3Pipeline --usp 8 --ring 1 --text-encoder-tp-size 8 --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 8 --enable-distributed-layerwise-offload --diffusion-attention-backend FLASH_ATTN --no-diffusion-compile-dynamic --stage-init-timeout 7200 --init-timeout 7200
python public_http_request.py --url http://127.0.0.1:18680 --images public-images --output "$NEW_OUTPUT/http-base"
```

Use `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, `OMNI_SOURCE_ROOT` and `PYTHONPATH` for the selected checkout; `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `MINDIE_SD_FA_TYPE=ascend_laser_attention`, `VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200`, `VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=7200`, and an independent `VLLM_OMNI_STORAGE_PATH` per server. Set a private `sitecustomize.py` directory first in `PYTHONPATH` that calls `torch.use_deterministic_algorithms(True)` only for the CLI/spawn workers; the attached `deterministic_sitecustomize.py` shows the exact condition. The original runner applied these same variables and used separate ports 18680/18681. The standalone HTTP client is a portable equivalent of its request loop, with identical request fields; it is not the original launch/cleanup harness. Recorded HTTP times are not performance measurements. Native staged H3 deployment was not tested in this HTTP check.

AI assistance: TRAE assisted the code, harness, validation and analysis.

## PR9468 GroupNorm/SiLU public image/video comparison

Candidate: `9323191e5305d66ef57ca73b049a65c5917756bb`, same main baseline,
public model, isolated runtime and component timing protocol as above.

| Input | Base mean ms | Candidate mean ms | Reduction | Exact arrays | Fused calls/request |
| --- | --- | --- | --- | --- | --- |
| Six public images | 2561.902 | 2539.494 | 0.875% | 30/30 | 1800 |
| Public video,39rawframes,8fps,832×480 | 7975.716 | 7925.052 | 0.635% | 5/5 | 900 |

Two warmups and three unprofiled calls, each arm/input in an independent
process, FP32 encoder and deterministic algorithms OFF. All pre/post physical
card/model-initializer admission checks passed. These are small component
savings; no HTTP or formal E2E gain,50-step quality or profile is claimed.
The earlier interrupted baseline-only attempts are excluded from this A/B.

Create the video array from the same SHA-verified public input:

```bash
ffmpeg -v error -i public-input.mp4 -vf 'fps=8,scale=832:480:flags=lanczos' -frames:v 39 -f rawvideo -pix_fmt rgb24 public-video.rgb
python -c 'import numpy as np; a=np.fromfile("public-video.rgb",dtype=np.uint8).reshape(39,480,832,3); np.save("public-video-39.npy",a,allow_pickle=False)'
```

NPY SHA256: `8c584283c110c89cc8e91fa969da239617cc4f96ba11b239aadcd9c83cccc35a`.
The source snapshot `validate_gn_media_component.py` is the exact measured
harness. Repeat the component commands with that script and PR9468 checkout;
add `--video-array "$PUBLIC_RUN_ROOT/public-video-39.npy"` for video,omit it
for images. The inherited result boundary label originally mentioned six
images for both branches; the video JSON records the original label and its
correction based on the actual video branch. Actual output shapes are retained;
raw39frames should not be interpreted as39untrimmed encoded frames.
