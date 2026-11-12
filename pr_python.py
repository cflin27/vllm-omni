"""Run PR code without the OCI image's editable checkout finders."""

import os
import runpy
import sys

sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if not getattr(finder, "__module__", "").startswith("__editable__")
]
sys.path[:] = [path for path in sys.path if not path.startswith("__editable__")]
source = os.environ["OMNI_SOURCE_ROOT"]
sys.path.insert(0, source)
os.chdir(source)
args = sys.argv[1:]
if args[0] == "-m":
    sys.argv = args[1:]
    runpy.run_module(args[1], run_name="__main__", alter_sys=True)
else:
    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")
