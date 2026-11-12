"""Small real NPU preflight for the dependency missing at native Stage0."""
import json

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch.npu.set_device(0)
assert enable_custom_op(), "Ascend custom operator library is unavailable"
generator = torch.Generator().manual_seed(17)
x = torch.randn(32, 128, generator=generator).to("npu:0")
residual = torch.randn(32, 128, generator=generator).to("npu:0")
weight = torch.ones(128, device="npu:0")
output, _, summed = torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, weight, None, 1e-6)
reference = (x.cpu() + residual.cpu())
reference = reference * torch.rsqrt(reference.square().mean(-1, keepdim=True) + 1e-6)
actual = output.cpu()
assert torch.isfinite(actual).all() and torch.allclose(actual, reference, atol=1e-5, rtol=1e-5)
assert torch.equal(summed.cpu(), x.cpu() + residual.cpu())
print(json.dumps({"operator": "npu_add_rms_norm_bias", "finite": True, "max_abs": (actual-reference).abs().max().item(),
                  "residual_exact": True, "torch": torch.__version__, "torch_npu": torch_npu.__version__}))
