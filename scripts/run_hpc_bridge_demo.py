"""Offline imported-result fixture. No SSH, cluster execution, or real OOM."""
import json
from pathlib import Path
from _execution import execution, definition
from ml_experiment_agent.db.models import Attempt, Run, Diagnosis, Metric


def write_fixture(folder: Path, *, failed: bool):
    manifest = json.loads((folder / "manifest.json").read_text())
    (folder / "stdout.log").write_text("Imported test fixture, not real LSF output\n")
    (folder / "stderr.log").write_text("torch.cuda.OutOfMemoryError: CUDA out of memory (fixture)\n" if failed else "")
    (folder / "exit.json").write_text(json.dumps({"attempt_id": manifest["attempt_id"], "exit_code": 1 if failed else 0}))
    if not failed:
        (folder / "checkpoint.pt").write_bytes(b"fixture only: not a real model; supplied metrics avoid checkpoint loading")
        (folder / "metadata.json").write_text('{"fixture":true}')
        (folder / "metrics.json").write_text('{"accuracy":0.8,"macro_f1":0.75,"ece":0.1}')


def main():
    root = Path(__file__).resolve().parents[1]
    print("HPC RESULTS / LLM / EMBEDDING = OFFLINE FIXTURES; memory unit MB is fixture configuration, not a site claim.")
    with execution(root, "hpc", memory_unit="MB", recovery=True) as (repo, service, graph):
        exp = service.create_experiment(definition(root, True))
        config = {"configurable": {"thread_id": exp.id}}
        graph.invoke({"experiment_id": exp.id, "user_goal": exp.goal}, config, durability="sync")
        run = repo.list(Run, experiment_id=exp.id)[0]
        first = repo.latest_attempt(run.id)
        first_folder = root / "data/exports" / exp.id / first.id
        print("Attempt 1 bundle:", first_folder)
        write_fixture(first_folder, failed=True)
        service.backend.import_result(first_folder)
        graph.invoke(None, config, durability="sync")
        second = repo.latest_attempt(run.id)
        assert first.status == "FAILED" and second.attempt_no == 2 and second.status == "QUEUED"
        assert first.config_json["batch_size"] == 128 and second.config_json["batch_size"] == 64
        second_folder = first_folder.parent / second.id
        print("Diagnosis:", repo.list(Diagnosis, attempt_id=first.id)[0].error_type)
        print("Attempt 2 bundle:", second_folder, "(waiting for human submission)")
        write_fixture(second_folder, failed=False)
        service.backend.import_result(second_folder)
        graph.invoke(None, config, durability="sync")
        saved = graph.get_state(config)
        assert saved.values["workflow_status"] == "SUCCEEDED"
        print("Final:", saved.values["workflow_status"], "Metrics:", {m.metric_name: m.metric_value for m in repo.list(Metric, attempt_id=second.id)})
        print("Report:", saved.values["report_path"])


if __name__ == "__main__":
    main()
