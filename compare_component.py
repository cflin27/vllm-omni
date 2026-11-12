"""Compare exported component arrays and unprofiled request times."""

import json
from pathlib import Path
import sys

import numpy as np

base, candidate = map(Path, sys.argv[1:3])
results = {}
for name, directory in [("base", base), ("candidate", candidate)]:
    results[name] = json.loads((directory / "result.json").read_text())
    results[name].pop("source_path", None)
parity = []
for path in sorted(base.glob("*.npy")):
    x, y = np.load(path, allow_pickle=False), np.load(candidate / path.name, allow_pickle=False)
    parity.append({"file": path.name, "exact": bool(np.array_equal(x, y)),
                   "max_abs": float(np.max(np.abs(x-y))),
                   "finite": bool(np.isfinite(x).all() and np.isfinite(y).all())})
results["parity"] = parity
results["improvement_percent"] = 100 * (1-results["candidate"]["mean_ms"] / results["base"]["mean_ms"])
print(json.dumps(results, indent=2))
if not parity or not all(row["exact"] and row["finite"] for row in parity):
    raise SystemExit(1)
