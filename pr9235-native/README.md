# PR9235 native serial/parallel public validation

Base `b7bbfbdb4f2b3c070d9632043ea6cd2b1c601270`; candidate
`508b2596d6c5b9b23e757a7937e3b6fb89ab384a`. Same public model revision,
six public images, eight Ascend 910B NPUs and isolated package/runtime versions
listed at the artifact root. Encoder TP8 and DiT USP8 are identical. The serial
decoder uses Video8 plus audio on its leader; the parallel decoder uses Video7
and Audio1 on disjoint devices. DLO, eager execution, FLASH_ATTN and expandable
segments are enabled in both arms. `HCCL_NPU_SOCKET_PORT_RANGE=auto` is identical.

| Native HTTP timing, ms | Request 1 | Request 2 | Request 3 | Arithmetic mean |
| --- | ---: | ---: | ---: | ---: |
| Serial | 10348.992 | 9816.270 | 9918.007 | 10027.756 |
| Parallel | 6970.831 | 7155.690 | 6991.524 | 7039.348 |

Observed reduction **29.801%** in this 1-NFE request, including MP4 generation.
Each arm has an independent server, one warmup and three consecutive unprofiled
requests with deterministic algorithms OFF. All samples are included. Repeated
reference images are used; cold or diverse-input latency, 50-NFE quality,
throughput and long-term stability were not measured. This is a single paired
run, without a reverse-order repeat or device profile.

Independent deterministic-ON servers each completed one warmup and one measured
precision request. Corresponding base/candidate MP4s are exact **2/2**, including
the warmup; measured precision is **1/1**. Decoded RGB24 and float32 PCM are also
exact for each corresponding pair. All 12 outputs across the timing/precision
runs decode successfully: 832×480, 124 video frames at 24 fps and stereo 32-kHz
audio. One sample pair does not establish 50-NFE model quality.

Warmup and measured precision outputs differ within each arm, identically in
base and candidate. The cause is unresolved; do not describe all four precision
MP4s as one identical file or claim universal request-to-request reproducibility.
`comparison.json` separates paired parity from self-repeat parity;
`media-comparison.json` retains the decoded differences and formulas.

All four native servers returned exit code 0 after parent SIGTERM, with no forced
worker cleanup. Final independent tracked/logged-PID audits found no matching
active processes. The performance baseline's original tracker saw only main-thread
children; its separate 26 logged-PID audit provides additional evidence, rather
than a claim of full historical start-time tracking. Other arms have all-thread
tracking/snapshots. Serial runs emitted resource-tracker shared-memory cleanup
warnings; parallel runs did not. No long-term leak-free claim is made.

`harness-performance.py` and `harness-precision.py` are the exact frozen programs
used for their respective arms. The difference traverses all task threads for
process diagnostics; request/config/timing behavior is the same. Results, media
specifications, admission checks, source hashes and C++ file hashes are attached.
Weights, large media files and raw machine logs are excluded.

## Reproduction

Create a fresh isolated native Ascend environment with the exact versions in
`runtime-package-versions.json`, CANN 9.1.0 and driver 25.5.0. The original OCI
manifest digest is `sha256:119485550d52238a7df54946efadb1a3a80daf3f784679f4e1068b7212fffd66`;
it required the listed package overlay and Ascend C++/vendor restoration.
The OCI image alone was insufficient for native Stage0 RMSNorm. Use public
vLLM-Ascend source `e617b709ff0a21fba34659f4f2f203b54699b236`, build/install its
C++ and vendor operators for `SOC_VERSION=ascend910b1` in that environment,
and apply the artifact's identical QSA 0.33 compatibility patch to both arms.
Verify the normal vendor installer succeeds and run `probe_rmsnorm.py` on an
admitted device before starting a full model server. The installed binary hashes
are recorded; binary files themselves are not distributed.

Use the root README's SHA-verified public-input/model preparation. The following
is a portable reproduction template, not the original absolute rootfs launcher.
`portable_run_omni_source_native.sh` selects an already configured native Python;
the measured launcher used the same source/instrument selection inside a relocated
OCI rootfs. Set these paths to new local directories:

```bash
export PUBLIC_ENV_ROOT="$PWD/native-validation"
export PUBLIC_RUN_ROOT="$PUBLIC_ENV_ROOT/validation"
export PUBLIC_PYTHON="/path/to/isolated/python3.12"
mkdir -p "$PUBLIC_RUN_ROOT"
cp -a /path/to/this/artifact/. "$PUBLIC_RUN_ROOT/"
cp "$PUBLIC_RUN_ROOT/portable_run_omni_source_native.sh" "$PUBLIC_ENV_ROOT/run_omni_source_native.sh"
chmod +x "$PUBLIC_ENV_ROOT/run_omni_source_native.sh"
# Put the pinned complete public model at:
# $PUBLIC_ENV_ROOT/public-model/MiniMax-H3
# Put the six generated PNGs at $PUBLIC_RUN_ROOT/public-images.
# Prepare two independent clean source checkouts at these exact commits.
export PUBLIC_BASE="/path/to/base-b7bbfbdb4"
export PUBLIC_CANDIDATE="/path/to/candidate-508b2596d"
export PUBLIC_DEVICES="0,1,2,3,4,5,6,7"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export ASCEND_RT_VISIBLE_DEVICES="$PUBLIC_DEVICES"
export OMNI_SOURCE_ROOT="$PUBLIC_BASE"
"$PUBLIC_PYTHON" "$PUBLIC_RUN_ROOT/pr9235-native/probe_rmsnorm.py"
"$PUBLIC_PYTHON" "$PUBLIC_RUN_ROOT/pr9235-native/harness-performance.py" --base "$PUBLIC_BASE" --candidate "$PUBLIC_CANDIDATE" --root "$PUBLIC_RUN_ROOT" --configs "$PUBLIC_RUN_ROOT/pr9235-native/configs" --devices "$PUBLIC_DEVICES" --repeat 3 --output "$PUBLIC_ENV_ROOT/new-performance"
"$PUBLIC_PYTHON" "$PUBLIC_RUN_ROOT/pr9235-native/harness-precision.py" --base "$PUBLIC_BASE" --candidate "$PUBLIC_CANDIDATE" --root "$PUBLIC_RUN_ROOT" --configs "$PUBLIC_RUN_ROOT/pr9235-native/configs" --devices "$PUBLIC_DEVICES" --deterministic --repeat 1 --output "$PUBLIC_ENV_ROOT/new-precision"
"$PUBLIC_PYTHON" "$PUBLIC_RUN_ROOT/compare_public_media.py" "$PUBLIC_ENV_ROOT/new-precision" --output "$PUBLIC_ENV_ROOT/new-media-comparison.json"
```

Source the configured CANN/ATB runtime environment before these commands. Verify
all eight physical devices and model initializers are free before and after each
arm; device visibility alone does not reserve hardware. Preserve source-local
version files and remove foreign editable-checkout finders with `pr_python.py`.
Use empty new output paths. Completion requires every request JSON, successful
full media decode, both arm `done` files, native exit records and final resource
admission. Compare corresponding request indices; exclude warmup/precision
requests from the timing mean. A new environment has not been rerun using this
portable launcher, so its portability is documented, not independently verified.

The missing-library and fixed-HCCL-port attempts completed no qualified timing
requests and are excluded. They are environment recovery evidence, not candidate
performance samples. No private inputs, prompts or model weights are included.
