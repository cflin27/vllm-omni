# H3 native video/audio stage parallel decoding

This experimental deployment keeps Encoder and DiT in their existing order and
fans the completed DiT latent payload out to independent Video and Audio stages.
The orchestrator joins both terminal results into the existing video response.
The feature is disabled unless `stage_parallel.mode: parallel` is explicit.

## Deployment configurations

Select a file in `vllm_omni/deploy/`:

| Suffix | Video devices | Audio devices | Required devices |
| --- | --- | --- | ---: |
| `_2plus1.yaml` | 0,1 | 3 | 4 (one reserved for decoding) |
| `_1plus1.yaml` | 0 | 3 | 4 (two reserved for decoding) |
| `_2plus2.yaml` | 0,1 | 2,3 | 4 (configuration example only) |
| `_3plus1.yaml` | 0,1,2 | 3 | 4 |
| `_4plus4.yaml` | 0,1,2,3 | 4,5,6,7 | 8 |
| `_7plus1.yaml` | 0,1,2,3,4,5,6 | 7 | 8 |

Current validation prioritizes **3+1, 2+1, 1+1**, followed by **7+1** when
eight-card testing begins. Audio currently has no multi-card compute speedup;
new multi-rank audio experiments are deferred.

The full prefix is `minimax_h3_disaggregated_parallel_decode`.
These examples retain the base Encoder/DiT placement. Changing a decoder
partition does not change the upstream model topology. Additional layouts can be
specified through stage `devices`, `parallel_config`, and `stage_parallel`.
Device counts must agree with each stage's parallel world size. Video uses tile
parallelism and `vae_patch_parallel_size` equal to its world size. Audio uses
`vae_patch_parallel_size: 1`, and **only its group leader loads and executes the
audio VAE**. Allocating more audio ranks currently provides no audio speedup.
Checkpoint, tile geometry, HCCL and memory constraints must be verified for every
layout; successful configuration validation alone does not prove execution.

With compatible vLLM, Omni and Ascend dependencies, the service command is:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m vllm_omni.entrypoints.cli.main serve /data1/weights/minimax-h3 \
  --omni --trust-remote-code --task-type fl2va \
  --deploy-config vllm_omni/deploy/minimax_h3_disaggregated_parallel_decode_3plus1.yaml \
  --host 127.0.0.1 --port 19098
```

The current remote environment cannot import this checkout's CLI because its
vLLM lacks `vllm.logger.configure_logging_from_args`. This command has **not**
passed complete API validation in that environment. Do not replace system
packages or modify an unrelated running service to work around this failure.

Keep experiment logs, temporary files and caches on the data disk. Start in an
isolated source checkout, record its archive hash and PID, and redirect stdout
and stderr to an experiment log. Poll `/health` before submitting requests. Stop
with SIGTERM sent to the recorded service PID, wait for its own children to
exit, and verify that its devices are released. Do not kill unrelated Python or
NPU processes.

## Cached latent validation

The following probe uses real `StageDiffusionClient`, worker processes,
`StagePool`, orchestrator fan-out, shared memory, terminal aggregation and
request cleanup. It injects a cached completed DiT boundary and does not load
Encoder or DiT:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python benchmarks/diffusion/minimax_h3_stage_parallel_probe.py \
  --model /data1/weights/minimax-h3 \
  --latent /path/to/video-latent.pt --audio-latent /path/to/audio-latent.pt \
  --video-devices 0,1,2 --audio-devices 3 \
  --out /data1/sp_probe/UNIQUE_NAME --warmups 2 --repeats 3
```

Use tensors from the same actual DiT request and preserve their provenance
(model revision, inference parameters, lossy settings, request/output index,
seed, dtype, shape, frame rate, duration). If `--audio-latent` is omitted, the
probe generates deterministic synthetic audio. Such results establish only
mechanics and must not be used for quality, A/V alignment or end-to-end RTF.

The default probe layout is 3+1. For 2+1 or 1+1, reduce only
`--video-devices` and keep one independent audio device.

The probe checks hashes, two **distinct** outputs and their order, cooperative
cancel/drain, timeout, an injected audio-only validation error, subsequent
request recovery, bindings and shared-memory cleanup. A recovery failure stops
the probe. Its timings include cached handoff through aggregation and exclude
Encoder, DiT and MP4 encoding. `result.json.completed` records the assertions;
inspect the process exit code independently. The known remote allocator failure
can occur after completed assertions, so `completed=true` does not establish a
clean shutdown or stability acceptance.

Use short cached inputs for capability screening. Before performance acceptance,
measure 3 warmups and 10 interleaved serial/parallel samples for the same layout,
then compare the selected layouts with the original full-device serial service
using the target output size and lossy settings. Use NPU profiler kernel
intervals to confirm actual computation overlap. Report complete-request
`RTF = end_to_end_seconds / video_duration_seconds`; decode-only times are not RTF.

## Failure handling and rollback

`stage_parallel.decode_timeout_s` is a positive finite timeout for fan-out through
both decoder terminals; its default is 120 seconds. Parallel mode admits one
whole request at a time and returns 429 for another request while it is active.
A branch failure or timeout aborts the request and drains active stages. Drain
waits for request tasks and device synchronization. A drain failure leaves the
service unavailable (503) until restart; an abort acknowledgement alone does
not prove that devices are idle. A device kernel or collective cannot always be
stopped immediately.

For rollback, stop the parallel service and restart with the existing
`minimax_h3_disaggregated_decode.yaml` configuration. Alternatively set
`stage_parallel.mode: serial` in a **copy** of the parallel deployment: the loader
selects the original three-stage pipeline and removes stage 3. That copy retains
stage 2's device list, so use the original deployment file to restore the
original full-device topology. No per-request hot switch is supported.

Incremental MP4 (`preencode_mp4=true`) remains on the serial pipeline. Parallel
stages accept complete decoding only. Complete API, incremental MP4 and default
serial regressions must pass before enabling the feature for production.
