import os
from pathlib import Path
import shutil
import subprocess
import sys
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.models import Diagnosis, Experiment, Attempt, AuditEvent


def test_fake_diagnosis_demo_end_to_end(environment):
    root, _, _ = environment
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / "knowledge", root / "knowledge")
    (root / "scripts").mkdir()
    script = root / "scripts/run_phase3a_diagnosis_demo.py"
    shutil.copyfile(source / "scripts/run_phase3a_diagnosis_demo.py", script)
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    env.update({"ML_AGENT_DB": "data/demo.sqlite", "HF_HUB_OFFLINE": "1", "LANGSMITH_TRACING": "false"})
    result = subprocess.run([sys.executable, str(script), "--fake"], cwd=root, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=90, check=True)
    assert "MODE=fake" in result.stdout and "not real E5" in result.stdout
    assert "diagnosis=CUDA_OOM" in result.stdout
    assert "read_job_log" in result.stdout and "retrieve_documentation" in result.stdout
    assert "COMPLETED_WITH_FAILURES" in result.stdout
    with create_session_factory(root / "data/demo.sqlite")() as session:
        repo = Repository(session)
        assert len(repo.list(Attempt)) == 3
        assert len(repo.list(Diagnosis)) == 1
        assert repo.list(Diagnosis)[0].evidence_json["mode"] == "fake"
        assert repo.list(Experiment)[0].status == "COMPLETED_WITH_FAILURES"
        assert repo.list(AuditEvent, action="diagnosis_completed")
