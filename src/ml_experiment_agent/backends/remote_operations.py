"""Fixed remote operations sent to python3 over SSH. No caller-provided commands.

This module uses only the standard library on the login node. Paths are resolved
there as well as locally, so existing symlinks cannot escape the configured root.
The remote project needs a prepared .venv and toy_project (see remote_lsf.md).
"""
import base64
import json
import math
from pathlib import Path
import re
import subprocess
import sys


def execute(request: dict) -> dict:
    root = Path(request["root"]).resolve(strict=True)
    if not root.is_dir() or root == Path("/") or root != Path(request["root"]).absolute():
        raise ValueError("Invalid remote project root")

    def path(relative: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", relative) or ".." in Path(relative).parts or Path(relative).is_absolute():
            raise ValueError("Invalid remote relative path")
        result = (root / relative).resolve()
        if not result.is_relative_to(root):
            raise ValueError("Remote symlink escapes project root")
        return result

    def command(args: list[str], cwd=None) -> dict:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=45, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout[-20000:], "stderr": result.stderr[-4000:]}

    operation = request["operation"]
    if operation == "check":
        required_paths = (("toy_project/lsf_smoke.py",) if request.get("smoke") else
                          (".venv/bin/python", "toy_project/train.py", "toy_project/evaluate.py"))
        for required in required_paths:
            if not path(required).is_file():
                raise ValueError("Remote project not prepared: " + required)
        return {"root": str(root), "ready": True}
    if operation == "submit":
        work = path(request["working"])
        entrypoint = path(request["entrypoint"])
        output = path(request["output"])
        output.mkdir(parents=True, exist_ok=True)
        config_path = path(request["output"] + "/config.yaml")
        config = request["config"]
        config["output_dir"] = str(output)
        # JSON is a YAML subset; no YAML library is required on the login node.
        config_path.write_text(json.dumps(config), encoding="utf-8")
        args = ["bsub", "-q", request["queue"], "-n", str(request["cpu"]),
                "-W", str(math.ceil(request["timeout"]))]
        if request["memory"] is not None:
            args += ["-R", f"rusage[mem={request['memory']}]"]
        elif request["entrypoint"] != "toy_project/lsf_smoke.py":
            raise ValueError("Cluster default memory is limited to the registered smoke probe")
        if request["gpu"]:
            args += ["-gpu", f"num={request['gpu']}:j_exclusive=yes"]
        interpreter = sys.executable if request["entrypoint"] == "toy_project/lsf_smoke.py" else str(path(".venv/bin/python"))
        args += ["-o", str(path(request["output"] + "/stdout.log")),
                 "-e", str(path(request["output"] + "/stderr.log")),
                 interpreter, str(entrypoint), "--config", str(config_path)]
        return command(args, cwd=work)
    if operation in ("status", "cancel"):
        job_id = request["job_id"]
        if not re.fullmatch(r"[1-9][0-9]*", job_id):
            raise ValueError("Invalid job ID")
        return command(["bjobs", "-a", "-noheader", "-o", "stat", job_id] if operation == "status" else ["bkill", job_id])
    if operation == "logs":
        tail = request["tail"]
        if type(tail) is not int or not 1 <= tail <= 200:
            raise ValueError("Invalid log tail")
        texts = []
        for name in ("stdout.log", "stderr.log"):
            log = path(request["output"] + "/" + name)
            if log.is_file():
                # Bound file I/O as well as returned text, even for huge logs.
                with log.open("rb") as handle:
                    handle.seek(max(0, log.stat().st_size - 65536))
                    texts.append("[" + name + "]")
                    texts.extend(handle.read(65536).decode("utf-8", errors="replace").splitlines()[-tail:])
        return {"text": "\n".join(texts[-tail:])[-32000:]}
    if operation == "artifact":
        if request["name"] not in ("checkpoint.pt", "metadata.json", "smoke_result.txt"):
            raise ValueError("Unsupported artifact")
        artifact = path(request["output"] + "/" + request["name"])
        if artifact.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Toy artifact exceeds 16 MiB transfer limit")
        return {"data": base64.b64encode(artifact.read_bytes()).decode("ascii")}
    raise ValueError("Unsupported remote operation")


if __name__ == "__main__":
    print(json.dumps(execute(json.load(sys.stdin))))
