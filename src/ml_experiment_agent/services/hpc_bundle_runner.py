"""Fixed exported runner. Run on HPC after manual bsub, never by an LLM."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    root = parser.parse_args().project_root.resolve(strict=True)
    folder = Path(__file__).resolve().parent
    if not folder.is_relative_to(root):
        raise ValueError("Bundle must be uploaded beneath the configured project root")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if json.loads((folder / "config.yaml").read_text(encoding="utf-8")) != manifest["config"]:
        raise ValueError("Bundle config was changed after validation")
    def registered(name):
        target = (root / manifest["working_directory"] / name).resolve(strict=True)
        if not target.is_relative_to(root):
            raise ValueError("Registered entrypoint escapes project root")
        return target
    config = manifest["config"] | {"output_dir": str(folder)}
    path = folder / "runtime_config.yaml"
    if (folder / "exit.json").exists():
        raise ValueError("Attempt already completed; export a new attempt instead")
    # Exclusive marker also blocks duplicate concurrent/manual resubmission.
    with (folder / "execution.started").open("x", encoding="utf-8") as marker:
        marker.write(manifest["attempt_id"])
    path.write_text(json.dumps(config), encoding="utf-8")
    code = 1
    with (folder / "stdout.log").open("wb") as stdout, (folder / "stderr.log").open("wb") as stderr:
        try:
            result = subprocess.run([sys.executable, str(registered(manifest["entrypoint"])), "--config", str(path)],
                cwd=root / manifest["working_directory"], stdout=stdout, stderr=stderr, shell=False, check=False)
            code = result.returncode
            if code == 0:
                result = subprocess.run([sys.executable, str(registered(manifest["eval_entrypoint"])), "--config", str(path)],
                    cwd=root / manifest["working_directory"], stdout=stdout, stderr=stderr, shell=False, check=False)
                code = result.returncode
        except (OSError, ValueError) as exc:
            code = 1
            stderr.write((type(exc).__name__ + ": " + str(exc)).encode())
        finally:
            (folder / "exit.json").write_text(json.dumps({"attempt_id": manifest["attempt_id"], "exit_code": code}), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
