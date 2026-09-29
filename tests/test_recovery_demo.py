import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest


@pytest.mark.parametrize("script,arguments,expected", [
    ("run_phase3b_recovery_demo.py", [], "SUCCEEDED"),
    ("run_phase3b_hitl_demo.py", [], "SUCCEEDED"),
    ("run_phase3b_hitl_demo.py", ["--reject"], "FAILED"),
])
def test_real_cpu_offline_recovery_demos(environment, script, arguments, expected):
    root, _, _ = environment
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / "knowledge", root / "knowledge")
    (root / "scripts").mkdir()
    for name in ("run_phase3b_recovery_demo.py", "run_phase3b_hitl_demo.py"):
        shutil.copy(source / "scripts" / name, root / "scripts" / name)
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    env.update(HF_HUB_OFFLINE="1", LANGSMITH_TRACING="false")
    completed = subprocess.run([sys.executable, str(root / "scripts" / script), *arguments], cwd=root,
                               env=env, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "MODE=fake" in completed.stdout and f"Final status={expected}" in completed.stdout
    assert "CUDA_OOM" in completed.stdout and "Attempt 1:" in completed.stdout
    if expected == "SUCCEEDED":
        assert "Attempt 2:" in completed.stdout and '"accuracy":' in completed.stdout
    else:
        assert "Attempt 2:" not in completed.stdout
    if "hitl" in script:
        assert "INTERRUPT=" in completed.stdout
    reports = list((root / "data/reports").glob("*.md"))
    assert len(reports) == 1 and "Recovery History" in reports[0].read_text(encoding="utf-8")
