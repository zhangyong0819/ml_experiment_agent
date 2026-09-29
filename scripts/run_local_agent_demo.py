"""Real processes; safe resource failure fixture, fake diagnosis/fix models."""
from pathlib import Path
import time
from _execution import execution, definition
from ml_experiment_agent.db.models import Run, Attempt, Metric


def main():
    root = Path(__file__).resolve().parents[1]
    print("BACKEND=local real subprocess; diagnosis/fix/RAG=offline fixture; no real OOM", flush=True)
    for recovery in (False, True):
        with execution(root, "local", recovery=recovery) as (repo, service, graph):
            exp = service.create_experiment(definition(root, recovery))
            config = {"configurable": {"thread_id": exp.id}}
            graph.invoke({"experiment_id": exp.id, "user_goal": exp.goal}, config, durability="sync")
            saved = graph.get_state(config)
            while saved.next == ("wait_for_poll",):
                time.sleep(.1)
                graph.invoke(None, config, durability="sync")
                saved = graph.get_state(config)
            print("Experiment", exp.id, "status", saved.values["workflow_status"], flush=True)
            for run in repo.list(Run, experiment_id=exp.id):
                attempts = sorted(repo.list(Attempt, run_id=run.id), key=lambda a: a.attempt_no)
                for attempt in attempts:
                    print("Attempt", attempt.attempt_no, attempt.status, "batch", attempt.config_json["batch_size"],
                          "job", attempt.job_id, "metrics", {m.metric_name: m.metric_value for m in repo.list(Metric, attempt_id=attempt.id)})
                assert len(attempts) == (2 if recovery else 1)
            print("Report", saved.values["report_path"], flush=True)
            assert saved.values["workflow_status"] == "SUCCEEDED"


if __name__ == "__main__":
    main()
