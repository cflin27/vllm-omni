"""Matched public-image HTTP correctness requests for an existing Omni PR."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import requests


def process_tree(pid, protected_patterns):
    records = []
    todo = [pid]
    while todo:
        current = todo.pop()
        entry = Path("/proc") / str(current)
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            command = (entry / "cmdline").read_bytes()
            if any(pattern.encode() in command for pattern in protected_patterns):
                raise RuntimeError("Protected process in experiment tree")
            records.append({"pid": current, "start_ticks": fields[19]})
            todo.extend(map(int, (entry / "task" / str(current) / "children").read_text().split()))
        except OSError:
            continue
    return records


def remaining(records, protected_patterns):
    result = []
    for record in records:
        entry = Path("/proc") / str(record["pid"])
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z" or fields[19] != record["start_ticks"]:
                continue
            if any(pattern.encode() in (entry / "cmdline").read_bytes() for pattern in protected_patterns):
                raise RuntimeError("Protected process cannot be signalled")
            result.append(record)
        except OSError:
            continue
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--configs", type=Path, required=True)
    p.add_argument("--devices", required=True)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--protected-process-pattern", action="append", default=[])
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = args.root.parent / "public-model/MiniMax-H3"
    images = sorted((args.root / "public-images").glob("*.png"))
    if len(images) != 6:
        raise ValueError("Expected exactly six public reference images")
    instrument = args.output / "instrument"
    instrument.mkdir()
    (instrument / "sitecustomize.py").write_text('''import sys
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, "__module__", "").startswith("__editable__")]
sys.path[:] = [p for p in sys.path if not p.startswith("__editable__")]
argv = " ".join(getattr(sys, "orig_argv", sys.argv))
if ENABLE_DETERMINISM and ("vllm_omni.entrypoints.cli.main" in argv or "multiprocessing.spawn" in argv):
    import torch
    torch.use_deterministic_algorithms(True)
'''.replace('ENABLE_DETERMINISM',str(args.deterministic)))
    for arm, source in [("base", args.base), ("candidate", args.candidate)]:
        d = args.output / arm; d.mkdir()
        port = 18880 + (arm == "candidate")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is in use")
        env = os.environ.copy()
        env.update(OMNI_SOURCE_ROOT=str(source), OMNI_INSTRUMENT_ROOT=str(instrument),
            HF_HOME=str(args.root / "hf-cache"), HF_MODULES_CACHE=str(args.root / "hf-modules"),
            ASCEND_RT_VISIBLE_DEVICES=args.devices, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            VLLM_OMNI_STORAGE_PATH=str(d / "storage"), PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
            VLLM_WORKER_MULTIPROC_METHOD="spawn", HCCL_NPU_SOCKET_PORT_RANGE="auto", VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS="15", MINDIE_SD_FA_TYPE="ascend_laser_attention",
            VLLM_OMNI_VIDEO_SYNC_TIMEOUT="7200", VLLM_OMNI_DLO_DP_WAVE_TIMEOUT="7200")
        command = [str(args.root.parent / "run_omni_source_native.sh"), "-B", str(args.root / "pr_python.py"),
            "-m", "vllm_omni.entrypoints.cli.main", "serve", str(model), "--omni", "--host", "127.0.0.1",
            "--port", str(port), "--trust-remote-code", "--task-type", "ref2va", "--num-gpus", "8",
            "--deploy-config",str(args.configs/(arm+".yaml")),
            "--enable-distributed-layerwise-offload", "--diffusion-attention-backend", "FLASH_ATTN",
            "--no-diffusion-compile-dynamic", "--stage-init-timeout", "7200", "--init-timeout", "7200"]
        (d / "launch.json").write_text(json.dumps({"command":command, "source_commit":subprocess.check_output(
            ["git","rev-parse","HEAD"],cwd=source,text=True).strip(), "scope":"native staged public1-NFE correctness" if args.deterministic else "native staged public1-NFE agile HTTP wall timing"},indent=2)+"\n")
        with (d / "server.log").open("w") as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (d / "server.pid").write_text(str(proc.pid)+"\n")
            owned_seen = {}
            try:
                deadline = time.monotonic() + 1200
                while time.monotonic() < deadline:
                    for item in process_tree(proc.pid, args.protected_process_pattern):
                        owned_seen[item["pid"]] = item
                    if proc.poll() is not None:
                        raise RuntimeError(f"{arm} server exited {proc.returncode}; see server.log")
                    log_path = d / "server.log"
                    with log_path.open("rb") as stream:
                        stream.seek(0, 2)
                        size = stream.tell()
                        stream.seek(max(0, size-65536))
                        tail = stream.read()
                    if b"StageEngineCoreProc failed to start." in tail:
                        raise RuntimeError(f"{arm} stage initialization failed; see server.log")
                    try:
                        if requests.get(f"http://127.0.0.1:{port}/health",timeout=2).ok:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(5)
                else:
                    raise TimeoutError(f"{arm} server health timeout")
                for label in ["warmup"]+[f"measure-{i+1}" for i in range(args.repeat)]:
                    data={"prompt":"Create a short realistic video using the scenes and person shown in <Picture 1>, <Picture 2>, <Picture 3>, <Picture 4>, <Picture 5>, and <Picture 6>, with natural motion and ambient sound.",
                        "width":"832", "height":"480", "fps":"24", "num_inference_steps":"1", "seed":"0", "flow_shift":"12.0",
                        "extra_params":json.dumps({"task":"ref2va", "duration":5.0,"audio_flow_shift":3.0})}
                    streams=[path.open("rb") for path in images]
                    try:
                        start=time.perf_counter()
                        response=requests.post(f"http://127.0.0.1:{port}/v1/videos/sync",data=data,
                            files=[("input_references",(path.name,stream,"image/png")) for path,stream in zip(images,streams)],timeout=7200)
                        elapsed=(time.perf_counter()-start)*1000
                    finally:
                        for stream in streams: stream.close()
                    response.raise_for_status()
                    if not response.headers.get("content-type","").startswith("video/mp4"):
                        raise ValueError("Expected video/mp4 response")
                    output=d/(label+".mp4");output.write_bytes(response.content)
                    decoded=subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(output),"-f","null","-"],capture_output=True,text=True)
                    decoded.check_returncode()
                    result={"label":label,"status":response.status_code,"wall_ms":elapsed,"bytes":len(response.content),
                        "sha256":hashlib.sha256(response.content).hexdigest(),"decode_rc":decoded.returncode}
                    (d/(label+".json")).write_text(json.dumps(result,indent=2)+"\n")
                    for item in process_tree(proc.pid, args.protected_process_pattern):
                        owned_seen[item["pid"]] = item
                    (d/"processes.json").write_text(json.dumps(list(owned_seen.values()),indent=2)+"\n")
                    print(arm,json.dumps(result),flush=True)
            finally:
                for item in process_tree(proc.pid, args.protected_process_pattern):
                    owned_seen[item["pid"]] = item
                owned = list(owned_seen.values())
                if proc.poll() is None:
                    # This session was created for this server alone. Never
                    # signal it if a protected process is unexpectedly present.
                    for entry in Path("/proc").iterdir():
                        if not entry.name.isdigit():
                            continue
                        try:
                            if os.getpgid(int(entry.name)) != proc.pid:
                                continue
                            command_line = (entry / "cmdline").read_bytes()
                        except (OSError, ProcessLookupError):
                            continue
                        if any(pattern.encode() in command_line for pattern in args.protected_process_pattern):
                            raise RuntimeError("Protected process found in server session")
                    os.kill(proc.pid,signal.SIGTERM)
                    try:
                        proc.wait(timeout=240)
                    except subprocess.TimeoutExpired:
                        pass
                leftover = remaining(owned, args.protected_process_pattern)
                forced = bool(leftover)
                if leftover:
                    for item in leftover:
                        os.kill(item["pid"], signal.SIGTERM)
                    deadline = time.monotonic() + 30
                    while remaining(owned, args.protected_process_pattern) and time.monotonic() < deadline:
                        time.sleep(2)
                    for item in remaining(owned, args.protected_process_pattern):
                        os.kill(item["pid"], signal.SIGKILL)
                    if proc.poll() is None:
                        proc.wait(timeout=30)
                (d/"exit.json").write_text(json.dumps({"server_exit_code":proc.returncode,
                    "parent_only_SIGTERM":not forced,"forced_cleanup":forced,
                    "remaining_pids":remaining(owned, args.protected_process_pattern)})+"\n")
        (d / "done").write_text("complete\n")


if __name__ == "__main__":
    main()
