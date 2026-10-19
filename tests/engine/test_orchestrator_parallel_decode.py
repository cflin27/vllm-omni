# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm_omni.engine.orchestrator import OrchestratorBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _harness(submits):
    orch = object.__new__(OrchestratorBase)
    orch.stage_pools = [None, None]
    for submit in submits:
        orch.stage_pools.append(
            SimpleNamespace(
                live_replica_ids=lambda: [0],
                submit_initial=submit,
                stage_client=SimpleNamespace(custom_process_input_func=lambda *args: {"prompt": "decode"}),
            )
        )
    orch._diffusion_submit_kwargs = Mock(return_value={})
    orch._build_payload_sender_info = Mock(return_value=None)
    orch._maybe_clone_diffusion_params_for_cfg = Mock(side_effect=lambda req_id, params: params)
    orch._on_stage_submitted = Mock()
    orch._emit_tx_edge = Mock()
    orch._cleanup_request_ids = AsyncMock()
    orch._fail_request_client_error = AsyncMock()
    state = SimpleNamespace(prompt={}, sampling_params_list=[None] * 4, stage_submit_ts={})
    return orch, state


@pytest.mark.asyncio
async def test_both_branches_submit_before_either_returns():
    entered = set()
    barrier = asyncio.Event()

    async def submit(stage_id, *args, **kwargs):
        entered.add(stage_id)
        if len(entered) == 2:
            barrier.set()
        await barrier.wait()
        return 0

    orch, state = _harness([lambda *a, **kw: submit(2, *a, **kw), lambda *a, **kw: submit(3, *a, **kw)])
    await asyncio.wait_for(orch._forward_to_fanout_stages("r", 1, object(), state, (2, 3)), 1)
    assert entered == {2, 3}
    assert set(state.stage_submit_ts) == {2, 3}
    orch._fail_request_client_error.assert_not_called()


@pytest.mark.asyncio
async def test_failed_submit_joins_cancelled_sibling_before_cleanup():
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def fail(*args, **kwargs):
        await entered.wait()
        raise RuntimeError("send failed")

    async def sibling(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    orch, state = _harness([fail, sibling])

    async def on_failure(*args, **kwargs):
        assert cancelled.is_set()

    orch._fail_request_client_error.side_effect = on_failure
    await asyncio.wait_for(orch._forward_to_fanout_stages("r", 1, object(), state, (2, 3)), 1)
    orch._fail_request_client_error.assert_awaited_once()
    assert not state.stage_submit_ts


@pytest.mark.asyncio
async def test_invalid_sibling_does_not_submit_first_branch():
    submit = AsyncMock(return_value=0)
    orch, state = _harness([submit, submit])
    orch.stage_pools[3].stage_client.custom_process_input_func = None
    await orch._forward_to_fanout_stages("r", 1, object(), state, (2, 3))
    submit.assert_not_called()
    orch._fail_request_client_error.assert_awaited_once()


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("order", [(2, 3), (3, 2)])
@pytest.mark.asyncio
async def test_real_formatter_outputs_join_in_either_order(batch, order):
    import numpy as np
    import torch

    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_decoder import (
        _audio_post_process,
        _video_post_process,
    )
    from vllm_omni.diffusion.output_formatter import format_diffusion_outputs, normalize_diffusion_postprocess_output
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.engine.orchestrator import OrchestratorRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    orch, _ = _harness([AsyncMock(), AsyncMock()])
    for sid in (2, 3):
        orch.stage_pools[sid].stage_client.result_aggregator = "minimax_h3_av"
        orch.stage_pools[sid].get_bound_replica_id = Mock(return_value=10 + sid)
    orch.stage_pools[0] = orch.stage_pools[1] = SimpleNamespace(stage_client=SimpleNamespace())
    orch.output_async_queue = asyncio.Queue()
    orch._cfg_tracker = SimpleNamespace(cleanup_parent=Mock(return_value=[]))
    state = OrchestratorRequestState(request_id="r", final_output_stage_ids={2}, stage_submit_ts={2: 1.0, 3: 2.0})
    audio_tensor = torch.arange(batch * 2 * 16).reshape(batch, 2, 16).float()
    values = {2: torch.zeros(batch, 3, 2, 2, 3, dtype=torch.uint8), 3: audio_tensor}
    outputs = {}
    for sid, postprocess, cls in (
        (2, _video_post_process, "MiniMaxH3VideoDecoderPipeline"),
        (3, _audio_post_process, "MiniMaxH3AudioDecoderPipeline"),
    ):
        raw = postprocess(values[sid])
        [outputs[sid]] = format_diffusion_outputs(
            request=OmniDiffusionRequest(
                prompt="p",
                request_id="r",
                sampling_params=OmniDiffusionSamplingParams(
                    num_outputs_per_prompt=batch,
                    num_inference_steps=1,
                ),
            ),
            od_config=SimpleNamespace(model_class_name=cls),
            diffusion_output=DiffusionOutput(output=values[sid], stage_durations={str(sid): float(sid)}),
            output_data=raw,
            postprocess_output=normalize_diffusion_postprocess_output(raw),
        )
    await orch._route_parallel_terminal_output(order[0], outputs[order[0]], state, {"stage": order[0]}, 0)
    assert orch.output_async_queue.empty()
    orch._cleanup_request_ids.assert_not_called()
    await orch._route_parallel_terminal_output(order[1], outputs[order[1]], state, {"stage": order[1]}, 0)
    message = orch.output_async_queue.get_nowait()
    assert message.stage_id == 2
    assert message.replica_id == 12
    assert message.finished
    assert message.metrics == {"stage": 2}
    result = message.engine_outputs
    assert len(result.images) == batch
    np.testing.assert_array_equal(result.multimodal_output["audio"], audio_tensor.numpy())
    assert result.multimodal_output["audio_sample_rate"] == 32000
    assert result.multimodal_output["fps"] == 24
    assert result.stage_durations == {"2": 2.0, "3": 3.0}
    orch._cleanup_request_ids.assert_awaited_once_with(["r"])


@pytest.mark.asyncio
async def test_duplicate_terminal_fails_request():
    from vllm_omni.engine.orchestrator import OrchestratorRequestState

    orch, _ = _harness([AsyncMock(), AsyncMock()])
    for sid in (0, 1):
        orch.stage_pools[sid] = SimpleNamespace(stage_client=SimpleNamespace())
    for sid in (2, 3):
        orch.stage_pools[sid].stage_client.result_aggregator = "minimax_h3_av"
    state = OrchestratorRequestState(request_id="r")
    output = SimpleNamespace(request_id="r")
    await orch._route_parallel_terminal_output(2, output, state, None, 0)
    await orch._route_parallel_terminal_output(2, output, state, None, 0)
    orch._fail_request_client_error.assert_awaited_once()


def test_process_diffusion_client_receives_routing_metadata():
    from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    metadata = SimpleNamespace(
        stage_id=2,
        replica_id=0,
        final_output=True,
        final_output_type="video",
        model_stage="video_vae",
        default_sampling_params=OmniDiffusionSamplingParams(),
        prompt_transform_func=None,
        prompt_expand_func=None,
        fanout_targets=(),
        input_payload_selector="video",
        result_aggregator="minimax_h3_av",
    )
    client = object.__new__(StageDiffusionClient)
    client._set_stage_metadata(metadata)
    assert client.result_aggregator == "minimax_h3_av"
    assert client.input_payload_selector == "video"
    assert client.fanout_targets == ()


@pytest.mark.asyncio
async def test_decode_deadline_fails_only_the_live_request():
    orch, state = _harness([AsyncMock(), AsyncMock()])
    state.parallel_cleanup_started = False
    orch.request_states = {"r": state}
    await orch._watch_parallel_decode_deadline("r", state, 0.001)
    assert orch._fail_request_client_error.await_args.kwargs["status_code"] == 504
    orch._fail_request_client_error.reset_mock()
    orch.request_states.clear()
    await orch._watch_parallel_decode_deadline("r", state, 0.001)
    orch._fail_request_client_error.assert_not_called()


@pytest.mark.parametrize("recoverable", [True, False])
@pytest.mark.asyncio
async def test_recovery_requires_idle_acknowledgement(recoverable):
    orch, _ = _harness([AsyncMock(), AsyncMock()])
    wait_idle = AsyncMock(side_effect=None if recoverable else TimeoutError("not idle"))
    client = SimpleNamespace(stage_id=2, wait_for_requests_idle_async=wait_idle, parallel_decode_timeout_s=0.1)
    await orch._drain_parallel_decode([client], ["r"])
    wait_idle.assert_awaited_once_with(["r"], 0.1)
    assert bool(getattr(orch, "_parallel_decode_recovery_error", None)) is not recoverable


@pytest.mark.parametrize("recovery_error,status", [(None, 429), ("device still busy", 503)])
@pytest.mark.asyncio
async def test_parallel_admission_stops_frontend_overlap(recovery_error, status):
    orch, _ = _harness([AsyncMock(), AsyncMock()])
    orch.stage_pools[2].stage_client.result_aggregator = "minimax_h3_av"
    orch.request_states = {"active": object()} if recovery_error is None else {}
    orch._parallel_decode_recovery_error = recovery_error
    orch.output_async_queue = asyncio.Queue()
    message = SimpleNamespace(
        request_id="next",
        prompt={},
        original_prompt={},
        sampling_params_list=[None] * 4,
        final_stage_id=3,
        final_output_stage_ids=[2, 3],
    )
    from vllm_omni.engine.orchestrator import Orchestrator

    await Orchestrator._handle_add_request(orch, message)
    result = orch.output_async_queue.get_nowait()
    assert result.status_code == status
    assert result.request_id == "next"
    orch.stage_pools[2].submit_initial.assert_not_called()


@pytest.mark.asyncio
async def test_abort_drains_bound_branch_without_submit_timestamp():
    from vllm_omni.engine.orchestrator import OrchestratorRequestState

    orch, _ = _harness([AsyncMock(), AsyncMock()])
    client = orch.stage_pools[2].stage_client
    client.result_aggregator = "minimax_h3_av"
    for sid in range(4):
        if sid < 2:
            orch.stage_pools[sid] = SimpleNamespace(stage_client=None)
        orch.stage_pools[sid].get_bound_client = Mock(return_value=client if sid == 2 else None)
    orch._cfg_tracker = SimpleNamespace(get_parent_id=Mock(return_value=None), cleanup_parent=Mock(return_value=[]))
    state = OrchestratorRequestState(request_id="r")
    orch.request_states = {"r": state}
    orch._pd_kv_params = {}
    orch._abort_request_ids = AsyncMock(return_value=[])
    orch._drain_parallel_decode = AsyncMock()
    orch._release_request_bindings = Mock()
    orch._release_stage_transfer_resources = Mock()
    await OrchestratorBase._cleanup_request_ids(orch, ["r"], abort=True)
    orch._drain_parallel_decode.assert_awaited_once_with([client], ["r"])
    assert not orch.request_states


@pytest.mark.asyncio
async def test_late_terminal_cannot_publish_success_during_abort():
    from vllm_omni.engine.orchestrator import OrchestratorRequestState

    orch, _ = _harness([AsyncMock(), AsyncMock()])
    orch._route_parallel_terminal_output = AsyncMock()
    state = OrchestratorRequestState(request_id="r", parallel_cleanup_started=True)
    await orch._route_output(2, 0, SimpleNamespace(request_id="r", finished=True), state, None)
    orch._route_parallel_terminal_output.assert_not_called()
