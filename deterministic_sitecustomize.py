import sys
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, "__module__", "").startswith("__editable__")]
sys.path[:] = [p for p in sys.path if not p.startswith("__editable__")]
argv = " ".join(getattr(sys, "orig_argv", sys.argv))
if "vllm_omni.entrypoints.cli.main" in argv or "multiprocessing.spawn" in argv:
    import torch
    torch.use_deterministic_algorithms(True)
