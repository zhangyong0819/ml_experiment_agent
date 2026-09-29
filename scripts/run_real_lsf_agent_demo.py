"""Explicit real SSH/LSF smoke. Passwords are entered only into local OpenSSH."""
import argparse
import json
import os
from pathlib import Path
import time
from ml_experiment_agent.backends.cluster import ClusterConfig
from ml_experiment_agent.backends.lsf import LSFBackend
from ml_experiment_agent.db.models import Run, Attempt, Metric
from ml_experiment_agent.db.repository import Repository
from ml_experiment_agent.db.session import create_session_factory
from ml_experiment_agent.graph.checkpoint import sqlite_checkpointer
from ml_experiment_agent.graph.dependencies import WorkflowDependencies, RunExecutionOptions
from ml_experiment_agent.graph.planning import FixturePlanner
from ml_experiment_agent.graph.workflow import build_experiment_graph
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ml_experiment_agent.models.project import ProjectRegistry
from ml_experiment_agent.services.experiment_service import ExperimentService
from ml_experiment_agent.services.policy_engine import PolicyEngine


def main(*, real_planner=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Allow native OpenSSH password/host-key prompts")
    parser.add_argument("--experiment-id", help="Resume an existing smoke checkpoint, never resubmit it")
    parser.add_argument("--wait-minutes", type=float, default=15)
    args = parser.parse_args()
    if os.getenv("ENABLE_REAL_LSF", "false").lower() != "true":
        raise SystemExit("SKIPPED: set ENABLE_REAL_LSF=true explicitly. No SSH command executed.")
    cluster = ClusterConfig.from_environment()
    if args.interactive:
        cluster.interactive_auth = True
    root = Path(__file__).resolve().parents[1]
    registry = ProjectRegistry(root / "project_specs", root)
    policy = PolicyEngine.load(root / "configs/policy.yaml", root, registry)
    database = root / "data/lsf_experiments.sqlite"
    checkpoints = root / "data/lsf_checkpoints.sqlite"
    factory = create_session_factory(database)
    try:
        with factory() as session:
            repo = Repository(session)
            backend = LSFBackend(policy, repo, cluster)
            print("Remote readiness:", backend.check_connection(), flush=True)
            if args.check_only:
                return
            service = ExperimentService(repo, registry, policy, backend)
            goal = "Run exactly one lightweight toy classification experiment, epochs=1, CPU only; compare accuracy, macro_f1, ece."
            definition = ExperimentDefinition(goal=goal, runs=[RunPlan(name="lsf_smoke",
                base_config=RunConfig(output_dir=root / "data/lsf_runs", epochs=1))])
            planner = FixturePlanner(definition)
            if real_planner:
                from ml_experiment_agent.llm.deepseek import create_deepseek_model
                from ml_experiment_agent.llm.planning import DeepSeekExperimentPlanner
                candidate = DeepSeekExperimentPlanner(create_deepseek_model(), root / "data/lsf_runs")
                class SmokePlanner:
                    def plan(self, goal, spec):
                        plan = candidate.plan(goal, spec)
                        if not 1 <= len(plan.runs) <= 2 or any(r.base_config.epochs > 2 or r.parameters.get("epochs", 1) > 2 for r in plan.runs):
                            raise ValueError("Real smoke budget is at most two runs of two epochs")
                        return plan
                planner = SmokePlanner()
            identifier = args.experiment_id or service.create_experiment(ExperimentDefinition(goal=goal)).id
            deps = WorkflowDependencies(service, planner, backend, default_execution_options=RunExecutionOptions(
                resources={"queue": cluster.queue, "cpu_count": 1, "gpu_count": 0, "memory_gb": 1, "timeout_minutes": 3}))
            config = {"configurable": {"thread_id": identifier}}
            print(f"MODE=real-lsf PLANNER={'deepseek' if real_planner else 'deterministic'} Experiment={identifier}", flush=True)
            with sqlite_checkpointer(checkpoints, application_db=database) as saver:
                graph = build_experiment_graph(deps, saver, deferred_polling=True)
                saved = graph.get_state(config)
                if args.experiment_id and not saved.values:
                    raise ValueError("No checkpoint exists for this experiment")
                initial = None if args.experiment_id else {"experiment_id": identifier, "user_goal": goal}
                if not saved.values or saved.next:
                    graph.invoke(initial, config, durability="sync")
                deadline = time.monotonic() + max(0, args.wait_minutes) * 60
                saved = graph.get_state(config)
                while saved.next == ("wait_for_poll",) and time.monotonic() < deadline:
                    for run in repo.list(Run, experiment_id=identifier):
                        attempt = repo.latest_attempt(run.id)
                        if attempt:
                            print(f"Job={attempt.job_id} status={attempt.status}", flush=True)
                    time.sleep(cluster.poll_interval_seconds)
                    graph.invoke(None, config, durability="sync")
                    saved = graph.get_state(config)
                for run in repo.list(Run, experiment_id=identifier):
                    for attempt in repo.list(Attempt, run_id=run.id):
                        print(f"Run={run.id} Attempt={attempt.id} Job={attempt.job_id} Status={attempt.status}")
                        if attempt.job_id:
                            print(backend.logs(attempt.job_id).text)
                        print("Metrics:", {m.metric_name: m.metric_value for m in repo.list(Metric, attempt_id=attempt.id)})
                print("Report:", saved.values.get("report_path"))
                print("Checkpoint DB:", checkpoints)
                if saved.next:
                    raise SystemExit(f"Still pending; checkpoint retained. Resume using --experiment-id {identifier}")
                if saved.values.get("workflow_status") != "SUCCEEDED":
                    raise SystemExit("Real validation did not succeed; inspect persisted logs/report.")
                result = {"experiment_id": identifier, "status": "SUCCEEDED", "real_lsf": True,
                          "planner": "deepseek" if real_planner else "deterministic", "report": saved.values["report_path"]}
                (root / "data/real_lsf_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                print("REAL LSF END-TO-END VALIDATED", flush=True)
    finally:
        factory.kw["bind"].dispose()


if __name__ == "__main__":
    main(real_planner=True)

