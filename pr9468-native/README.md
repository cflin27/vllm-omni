# PR9468 current-head native end-to-end accuracy

Base `b7bbfbdb4f2b3c070d9632043ea6cd2b1c601270`; GN/SiLU candidate
`9323191e5305d66ef57ca73b049a65c5917756bb`. Both arms use identical native
three-stage configs: Encoder TP8, DiT USP8 and serial Video8/audio-leader decode.
Only the GN/SiLU PR differs. The same public model revision, six SHA-verified
official-video-derived images and isolated runtime are used. Physical devices
0–7 passed complete-group admission before/after the run. DLO, eager,
FLASH_ATTN, expandable segments and dynamic HCCL ports match both arms.

Each arm has an independent deterministic-ON server, one warmup and one measured
precision request: 832×480, 24fps, duration5, 1NFE, seed0, flow_shift12 and
audio_flow_shift3. This is accuracy validation; deterministic HTTP wall times
are retained as raw records and excluded from performance claims.

| Corresponding requests | MP4 exact | RGB24 exact | Float32 PCM exact |
| --- | --- | --- | --- |
| Warmup base/candidate | 1/1 | 1/1 | 1/1 |
| Measured base/candidate | 1/1 | 1/1 | 1/1 |

Maximum decoded video/audio difference is0 for each corresponding pair. All four
MP4s fully decode, with124 video frames at24fps, 832×480, stereo32kHz audio.
Within each arm, warmup and measured media differ identically in base/candidate;
the cause is unresolved. Paired precision and self-repeat are reported separately.
These observations cover one measured public image-conditioning case at1NFE.
50-NFE quality, native video-conditioning E2E, throughput and long-term stability
were not run. The separate public image/video component tests establish the
recorded fused-call counts and small component savings; this run adds no E2E
performance result.

Both native parents exited0 after parent SIGTERM, with no forced cleanup. Final
tracked/logged-PID audits found no matching active processes. Both serial servers
emitted resource-tracker shared-memory cleanup warnings. All84 recorded CLI/spawn
startup deterministic states were enabled; these are startup observations.

`harness.py` is the frozen executed program, with all-task-thread process tracking
and deterministic startup records. The JSON includes every request/hash, source
commit, media formula/specification and qualified exit audit. Weights, large
media and machine logs are excluded.

## Reproduction

Use the public input/model preparation and configured isolated native environment
from the root README and [PR9235 environment steps](../pr9235-native/README.md).
The portable launcher requires already installed CANN9.1/Ascend C++/vendor support;
the exact binary/package versions are attached there. Against two fresh clean
source checkouts at the above commits, use a new output directory:

```bash
export PUBLIC_BASE="/path/to/base-b7bbfbdb4"
export PUBLIC_CANDIDATE="/path/to/candidate-9323191e5"
"$PUBLIC_PYTHON" "$PUBLIC_RUN_ROOT/pr9468-native/harness.py" --base "$PUBLIC_BASE" --candidate "$PUBLIC_CANDIDATE" --root "$PUBLIC_RUN_ROOT" --configs "$PUBLIC_RUN_ROOT/pr9468-native/configs" --devices "$PUBLIC_DEVICES" --deterministic --repeat 1 --output "$PUBLIC_ENV_ROOT/new-gn-precision"
"$PUBLIC_PYTHON" "$PUBLIC_RUN_ROOT/compare_public_media.py" "$PUBLIC_ENV_ROOT/new-gn-precision" --output "$PUBLIC_ENV_ROOT/new-gn-media-comparison.json"
```

Keep the two YAMLs identical. Check complete physical-device/model-initializer
admission before and after each arm. Require HTTP200, full FFmpeg decode, both
`done` markers, source/config hashes, normal native exit and process audit.
Compare corresponding request indices and retain self-repeat separately. Native
portable-environment reconstruction remains an unexecuted template; the recorded
local isolated run completed. The public component scope is documented separately
at the artifact root.
