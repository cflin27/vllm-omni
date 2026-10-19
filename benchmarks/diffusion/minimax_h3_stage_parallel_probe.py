# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cached DiT boundary -> real Orchestrator/StagePool -> native NPU stages.
Synthetic audio tests mechanics only; this is not a complete API/RTF test.
"""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--video-devices", default="0,1,2")
    parser.add_argument("--audio-devices", default="3")
    parser.add_argument("--latent", default="/data1/sp_probe/s26_square_resolution_20261015/latent.pt")
    parser.add_argument("--audio-latent", help="Optional cached audio tensor matching the video DiT output")
    parser.add_argument("--model", default="/data1/weights/minimax-h3")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats must be positive")
    device_groups = []
    for name in ("video_devices", "audio_devices"):
        try:
            ids = [int(value.strip()) for value in getattr(args, name).split(",")]
        except ValueError:
            parser.error(f"{name} must be a nonempty comma-separated list of device IDs")
        if not ids or any(value < 0 for value in ids) or len(set(ids)) != len(ids):
            parser.error(f"{name} must contain distinct nonnegative device IDs")
        setattr(args, name, ",".join(map(str, ids)))
        device_groups.append(set(ids))
    if device_groups[0] & device_groups[1]:
        parser.error("this probe requires disjoint video and audio device groups")

    def interrupted(signum, frame):
        raise TimeoutError(f"probe interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)
    status = {
        "scope": "cached_DiT_boundary_real_orchestrator_NOT_complete_API_or_RTF",
        "video_devices": args.video_devices,
        "audio_devices": args.audio_devices,
        "video_source": args.latent,
        "audio_source": "synthetic_seed42_MECHANISM_ONLY",
        "runs": [],
        "faults": [],
        "completed": False,
    }
    clients = []

    def emit(event, **values):
        print(json.dumps({"event": event, "monotonic": time.monotonic(), **values}, default=str), flush=True)

    try:
        import numpy as np
        import torch

        from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
        from vllm_omni.diffusion.stage_diffusion_client import create_diffusion_client
        from vllm_omni.engine.orchestrator import OrchestratorBase, OrchestratorRequestState
        from vllm_omni.engine.stage_init_utils import StageMetadata
        from vllm_omni.engine.stage_pool import StagePool
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        from vllm_omni.model_executor.stage_input_processors.minimax_h3 import (
            diffusion2audio_decoder,
            diffusion2video_decoder,
        )
        from vllm_omni.outputs import OmniRequestOutput

        video = torch.load(args.latent, map_location="cpu", weights_only=True)
        audio_cfg = json.loads((Path(args.model) / "FL2VA/audio_vae/config.json").read_text())
        audio = (
            torch.load(args.audio_latent, map_location="cpu", weights_only=True)
            if args.audio_latent
            else torch.randn((1, int(audio_cfg["latent_channels"]), 32), generator=torch.Generator().manual_seed(42))
        )
        if args.audio_latent:
            status["audio_source"] = args.audio_latent
        status["latent_dtypes"] = {"video": str(video.dtype), "audio": str(audio.dtype)}
        status["provenance"] = {
            "video_model_commit": "unknown_in_imported_cache",
            "synthetic_audio_seed": None if args.audio_latent else 42,
            "quality_validation": "requires paired real DiT cache and full provenance",
        }
        height, width = int(video.shape[-2]) * 16, int(video.shape[-1]) * 16
        status["latent_shapes"] = {"video": list(video.shape), "audio": list(audio.shape)}
        status["input_sha256"] = {
            m: hashlib.sha256(v.numpy().tobytes()).hexdigest() for m, v in [("video", video), ("audio", audio)]
        }
        for modality, devices, sid, processor in [
            ("video", args.video_devices, 2, diffusion2video_decoder),
            ("audio", args.audio_devices, 3, diffusion2audio_decoder),
        ]:
            world = len(devices.split(","))
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = devices
            config = OmniDiffusionConfig(
                stage_id=sid,
                model=args.model,
                model_class_name=f"MiniMaxH3{modality.title()}DecoderPipeline",
                task_type="fl2va",
                trust_remote_code=True,
                num_gpus=world,
                distributed_executor_backend="mp",
                dist_timeout=60,
                parallel_config=DiffusionParallelConfig(
                    ulysses_degree=world,
                    vae_patch_parallel_size=world if modality == "video" else 1,
                    vae_parallel_mode="tile",
                ),
                vae_use_tiling=True,
            )
            metadata = StageMetadata(
                stage_id=sid,
                stage_type="diffusion",
                engine_output_type=modality,
                is_comprehension=False,
                requires_multimodal_data=False,
                engine_input_source=[1],
                final_output=True,
                final_output_type=modality,
                default_sampling_params={},
                custom_process_input_func=processor,
                model_stage=f"{modality}_decoder",
                runtime_cfg={"parallel_decode_timeout_s": 60.0},
                result_aggregator="minimax_h3_av",
                input_payload_selector=modality,
            )
            emit("starting_stage", stage_id=sid, devices=devices, world=world)
            clients.append(create_diffusion_client(args.model, config, metadata, 150, use_inline=False))
            emit("stage_ready", stage_id=sid)

        async def experiments():
            pools = [StagePool(0, []), StagePool(1, []), StagePool(2, clients[0]), StagePool(3, clients[1])]
            output_queue = asyncio.Queue()
            orch = OrchestratorBase(asyncio.Queue(), output_queue, asyncio.Queue(), pools)
            baseline_hashes = None
            shm_start = set(Path("/dev/shm").glob("h3_latent_*"))
            status["shm_initial_count"] = len(shm_start)

            async def released():
                tasks = list(orch._transfer_release_tasks)
                if tasks:
                    await asyncio.gather(*tasks)

            async def poll_once():
                for sid, client in zip((2, 3), clients):
                    # Buffered scheduler/start notifications can precede terminals.
                    for _ in range(16):
                        result = client.get_diffusion_output_nowait()
                        if result is None:
                            break
                        state = orch.request_states.get(result.request_id)
                        if state is not None:
                            await orch._route_output(sid, 0, result, state, None)
                        else:
                            emit("discard_late_output", stage_id=sid, request_id=result.request_id)

            async def finish(request_id):
                limit = time.monotonic() + 90
                while time.monotonic() < limit:
                    await poll_once()
                    if not output_queue.empty():
                        msg = output_queue.get_nowait()
                        assert msg.request_id == request_id, (msg.request_id, request_id)
                        # A deadline task emits the error before drain completes.
                        while request_id in orch.request_states and time.monotonic() < limit:
                            await asyncio.sleep(0.005)
                            await poll_once()
                        assert request_id not in orch.request_states, "cleanup did not finish"
                        await released()
                        return msg
                    await asyncio.sleep(0.002)
                raise TimeoutError(f"orchestrator probe timed out: {request_id}")

            def inputs(request_id, batch=1, alternate=False):
                params = OmniDiffusionSamplingParams(seed=42, height=height, width=width, num_outputs_per_prompt=batch)
                state = OrchestratorRequestState(
                    request_id=request_id,
                    prompt={"prompt": "", "additional_information": {"global_request_id": request_id}},
                    sampling_params_list=[params] * 4,
                    final_stage_id=3,
                    final_output_stage_ids={2, 3},
                    finished_stage_ids={0, 1},
                )
                orch.request_states[request_id] = state
                source = OmniRequestOutput(
                    request_id=request_id,
                    finished=True,
                    latents={
                        "video": [video * 0.9 if alternate or j else video for j in range(batch)],
                        "audio": [audio * 0.9 if alternate or j else audio for j in range(batch)],
                    },
                    _multimodal_output={
                        "metadata": {
                            "minimax_h3_decode": {
                                "height": height,
                                "width": width,
                                "preencode_mp4": False,
                                "fps": 24,
                                "audio_sample_rate": 32000,
                            }
                        }
                    },
                )
                return state, source

            async def assert_clean():
                await released()
                assert not orch.request_states
                assert all(not p._request_bindings for p in pools)
                assert all(not c._latent_transport.pending for c in clients)
                assert not getattr(orch, "_parallel_decode_recovery_error", None), getattr(
                    orch, "_parallel_decode_recovery_error", None
                )
                new_shm = set(Path("/dev/shm").glob("h3_latent_*")) - shm_start
                assert not new_shm, [str(p) for p in new_shm]

            alternate_hashes = None

            async def run(index, batch=1, tag="parallel", alternate=False):
                nonlocal baseline_hashes, alternate_hashes
                rid = f"orch-{tag}-{index}"
                state, source = inputs(rid, batch=batch, alternate=alternate)
                started = time.monotonic()
                await orch._forward_to_fanout_stages(rid, 1, source, state, (2, 3))
                msg = await finish(rid)
                assert msg.type == "output", repr(msg)
                result = msg.engine_outputs
                arrays = {"video": np.asarray(result.images), "audio": np.asarray(result.multimodal_output["audio"])}
                assert all(np.isfinite(v).all() for v in arrays.values()), "nonfinite decoder output"
                hashes = {m: hashlib.sha256(v.tobytes()).hexdigest() for m, v in arrays.items()}
                if alternate:
                    alternate_hashes = hashes
                    assert all(hashes[m] != baseline_hashes[m] for m in hashes), "order test needs distinct outputs"
                elif batch == 1:
                    if baseline_hashes is None:
                        baseline_hashes = hashes
                    assert hashes == baseline_hashes
                else:
                    for m, values in arrays.items():
                        assert len(values) == batch
                        assert hashlib.sha256(values[0:1].tobytes()).hexdigest() == baseline_hashes[m]
                        assert hashlib.sha256(values[1:2].tobytes()).hexdigest() == alternate_hashes[m]
                assert msg.stage_id == 2
                row = {
                    "request_id": rid,
                    "seconds": time.monotonic() - started,
                    "index": index,
                    "warmup": index < args.warmups,
                    "batch": batch,
                    "shapes": {m: list(v.shape) for m, v in arrays.items()},
                    "sha256": hashes,
                }
                status["runs"].append(row)
                emit("run_complete", **row)
                await assert_clean()

            for index in range(args.warmups + args.repeats):
                await run(index)
            await run(args.warmups + args.repeats, tag="alternate-single", alternate=True)
            await run(args.warmups + args.repeats, batch=2, tag="multi-output")

            for fault in ("cancel", "timeout", "branch_error"):
                rid = f"orch-{fault}"
                state, source = inputs(rid)
                # Inject a request-scoped validation failure into audio only.
                # Both real processors, stage dispatch and aggregation remain in use.
                if fault == "branch_error":

                    def invalid_audio(*args, **kwargs):
                        prompt = diffusion2audio_decoder(*args, **kwargs)
                        prompt["additional_information"]["minimax_h3_decode"]["preencode_mp4"] = True
                        return prompt

                    clients[1].custom_process_input_func = invalid_audio
                if fault == "timeout":
                    for client in clients:
                        client.parallel_decode_timeout_s = 0.02
                started = time.monotonic()
                await orch._forward_to_fanout_stages(rid, 1, source, state, (2, 3))
                # Restore recovery's idle wait budget; deadline already captured .02.
                for client in clients:
                    client.parallel_decode_timeout_s = 60.0
                clients[1].custom_process_input_func = diffusion2audio_decoder
                if fault == "cancel":
                    await asyncio.sleep(0.02)
                    await orch._cleanup_request_ids([rid], abort=True)
                    await poll_once()
                    assert output_queue.empty(), "cancelled request published a response"
                else:
                    msg = await finish(rid)
                    assert msg.type == "error", repr(msg)
                    if fault == "timeout":
                        assert msg.status_code == 504, repr(msg)
                await assert_clean()
                row = {"kind": fault, "seconds": time.monotonic() - started, "recovered": True}
                status["faults"].append(row)
                emit("fault_complete", **row)
                await run(args.warmups + args.repeats, tag=f"after-{fault}")
            status["shm_final_count"] = len(list(Path("/dev/shm").glob("h3_latent_*")))
            status["output_hashes"] = baseline_hashes

        asyncio.run(experiments())
        status["completed"] = True
    except BaseException as exc:
        status["error"] = repr(exc)
        import traceback

        traceback.print_exc()
        raise
    finally:
        for client in reversed(clients):
            try:
                client.shutdown()
            except Exception as exc:
                emit("shutdown_error", error=repr(exc))
        (directory / "result.json").write_text(json.dumps(status, indent=2, default=str))
        emit("probe_finished", completed=status["completed"])


if __name__ == "__main__":
    main()
