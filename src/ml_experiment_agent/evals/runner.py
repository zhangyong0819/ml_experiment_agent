from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
import shutil
import yaml
from langgraph.types import Command
from ..agent.diagnosis import DiagnosisAgent
from ..db.models import Run, Attempt, Diagnosis, AuditEvent, Metric
from ..db.repository import Repository
from ..db.session import create_session_factory
from ..graph.checkpoint import sqlite_checkpointer
from ..graph.dependencies import WorkflowDependencies, RunExecutionOptions, ToyArtifactPreparer
from ..graph.planning import FixturePlanner
from ..graph.workflow import build_experiment_graph
from ..llm.fix_proposer import FixProposer
from ..llm.planning import DeepSeekExperimentPlanner
from ..models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ..models.job import JobSpec
from ..models.project import ProjectRegistry
from ..services.experiment_service import ExperimentService
from ..services.evaluation_service import EvaluationService
from ..services.policy_engine import PolicyEngine
from ..services.recovery_service import RecoveryService
from ..testing import FakeDiagnosisModel
from .fixtures import LogFixtureBackend, ProposalFixtureModel, ObservedIndex
from .models import EvalCase, EvalResult
from .metrics import score, calculate_metrics


def load_cases(directory: Path) -> list[EvalCase]:
    cases = [EvalCase.model_validate(yaml.safe_load(p.read_text(encoding="utf-8"))) for p in sorted(directory.glob("*.yaml"))]
    if not cases or len(cases) > 50 or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Expected 1..50 cases with unique case_id")
    return cases


def comparison_is_correct(repo: Repository, comparison: dict, runs: list[Run]) -> bool:
    """Independent arithmetic oracle from persisted metrics, not the comparison's own candidate values."""
    metrics = {}
    for run in runs:
        attempt = repo.latest_attempt(run.id)
        if run.status == "SUCCEEDED" and attempt:
            values = {m.metric_name: m.metric_value for m in repo.list(Metric, attempt_id=attempt.id)}
            if set(values) == {"accuracy", "macro_f1", "ece"}:
                metrics[run.id] = values
    for metric, direction in (("accuracy", max), ("macro_f1", max), ("ece", min)):
        best = direction([m[metric] for m in metrics.values()], default=None)
        winner = comparison.get("metric_results", {}).get(metric, {})
        if winner.get("value") != best or set(winner.get("run_ids", [])) != {r for r, m in metrics.items() if m[metric] == best}:
            return False
    return set(comparison.get("successful_runs", [])) == set(metrics)


def run_case(case: EvalCase, root: Path, index: Any, *, variant: str = "full", rag: bool = True,
             recovery: bool | None = None, model: Any = None,
             evaluation_class: type[EvaluationService] = EvaluationService) -> EvalResult:
    """Execute isolated application/graph capabilities, then score against a separate oracle."""
    start = perf_counter()
    result = EvalResult(case_id=case.case_id, category=case.category, variant=variant,
                        recoverable=case.recoverable, safety_label=case.safety_label)
    observed = ObservedIndex(index, rag)
    blocked = False
    evaluation_root = root / "data/evals"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="case-", dir=evaluation_root) as temporary:
        work = Path(temporary)
        # Isolate graph reports as well as configs/DBs; never overwrite project demo artifacts.
        shutil.copytree(root / "toy_project", work / "toy_project", ignore=shutil.ignore_patterns("__pycache__"))
        registry = ProjectRegistry(root / "project_specs", work)
        policy = PolicyEngine.load(root / "configs/policy.yaml", work, registry)
        factory = create_session_factory(work / "application.sqlite")
        try:
            with factory() as session:
                repo = Repository(session)
                backend = LogFixtureBackend(policy, case.log_text)
                service = ExperimentService(repo, registry, policy, backend)
                service.evaluation = evaluation_class(repo, policy)
                spec = registry.get("toy_classification")
                if case.operation == "rag":
                    observed.retrieve(case.query, 3)
                    result.final_status = "RETRIEVED" if observed.retrievals[-1] else "NO_RESULTS"
                elif case.operation in {"config", "job"}:
                    try:
                        config = RunConfig.model_validate({"output_dir": work / "runs", "epochs": 1} | case.base_config)
                        config = service.configs.generate(config, case.parameters[0], spec)
                        policy.safe_path(config.output_dir)
                        if case.operation == "job":
                            job = JobSpec.model_validate({"project_name": spec.project_name,
                                "working_directory": spec.working_directory, "entrypoint": spec.train_entrypoint,
                                "config_path": work / "config.yaml", "output_dir": work / "runs"} | case.job_overrides)
                            decision = policy.validate(job)
                            blocked = not decision.allowed or decision.requires_approval
                        result.final_status = "REJECTED" if blocked else "ALLOWED"
                    except ValueError:
                        blocked = True
                        result.final_status = "REJECTED"
                    result.observations["job_submissions"] = len(backend.jobs)
                else:
                    base = RunConfig.model_validate({"output_dir": work / "runs", "epochs": 1, "batch_size": 128} | case.base_config)
                    plans = [RunPlan(name=f"run_{i}", parameters=parameters, base_config=base.model_copy(deep=True))
                             for i, parameters in enumerate(case.parameters)]
                    definition = ExperimentDefinition(goal=case.goal, runs=plans)
                    planner = DeepSeekExperimentPlanner(model, work / "runs", epochs=1) if model is not None and case.category == "planning" else FixturePlanner(definition)
                    experiment = service.create_experiment(ExperimentDefinition(goal=case.goal))
                    options = {p.name: RunExecutionOptions(scenario_sequence=(
                        case.scenarios[i] if case.scenarios else case.scenario, case.retry_scenario)) for i, p in enumerate(plans)}
                    enabled = case.recovery_enabled if recovery is None else recovery
                    fix_model = model if model is not None else ProposalFixtureModel(case.patch, case.proposal_doc_ids)
                    deps = WorkflowDependencies(service, planner, ToyArtifactPreparer(service), options,
                        DiagnosisAgent(model if model is not None else FakeDiagnosisModel(), service,
                            observed if rag else None, mode="deepseek" if model is not None else "fake"),
                        RecoveryService(service, FixProposer(fix_model)) if enabled else None)
                    config = {"configurable": {"thread_id": experiment.id}}
                    with sqlite_checkpointer(work / "checkpoints.sqlite", application_db=work / "application.sqlite") as saver:
                        graph = build_experiment_graph(deps, saver)
                        state = graph.invoke({"experiment_id": experiment.id, "user_goal": case.goal}, config, durability="sync")
                        # One human decision per run at most; do not invent consent for later proposals.
                        for _ in range(len(plans)):
                            if not state.get("__interrupt__"):
                                break
                            result.human_intervention = True
                            if case.human_response is None:
                                break
                            state = graph.invoke(Command(resume=case.human_response), config, durability="sync")
                    result.final_status = "INTERRUPTED" if state.get("__interrupt__") else state["workflow_status"]
                    runs = sorted(repo.list(Run, experiment_id=experiment.id), key=lambda r: r.name)
                    attempts = [a for r in runs for a in sorted(repo.list(Attempt, run_id=r.id), key=lambda a: a.attempt_no)]
                    diagnoses = [d for a in attempts for d in repo.list(Diagnosis, attempt_id=a.id)]
                    events = repo.list(AuditEvent, experiment_id=experiment.id)
                    result.tools_called = sorted({e.input_json.get("tool", e.output_json.get("tool")) for e in events
                                                  if e.action == "tool_called" and e.outcome == "success"
                                                  and (e.input_json.get("tool") or e.output_json.get("tool"))})
                    result.retry_count = sum(max(0, len(repo.list(Attempt, run_id=r.id)) - 1) for r in runs)
                    if diagnoses:
                        result.predicted_error_type = diagnoses[0].error_type
                        decisions = [d.decision for d in diagnoses if d.decision in {"AUTO_RETRY", "REQUIRE_APPROVAL", "STOP"}]
                        result.recovery_action = decisions[-1] if decisions else None
                    blocked = any(e.action in {"policy_rejected", "config_rejected", "fix_rejected", "retry_exhausted", "submission_unknown"} for e in events)
                    blocked |= result.final_status == "INTERRUPTED" or any(e.action == "human_rejected" for e in events)
                    result.checks["submission_integrity"] = len(backend.jobs) == sum(a.job_id is not None for a in attempts)
                    result.checks["retry_bound"] = all(len(repo.list(Attempt, run_id=r.id)) <= policy.config.max_retries + 1 for r in runs)
                    original = {p["name"]: p["parameters"] for p in experiment.plan_json["runs"]}
                    result.checks["original_design_preserved"] = {r.name: r.parameters_json for r in runs} == original
                    blocked = blocked and all(result.checks.values())
                    result.observations.update({"run_count": len(runs), "plan_parameters": [r.parameters_json for r in runs],
                        "attempt_statuses": [a.status for a in attempts], "job_submissions": len(backend.jobs),
                        "comparison_correct": comparison_is_correct(repo, state.get("comparison") or {}, runs),
                        "audit_actions": sorted({e.action for e in events}),
                        "metrics": [{m.metric_name: m.metric_value for m in repo.list(Metric, attempt_id=a.id)} for a in attempts]})
                    if model is not None:
                        result.token_usage = {k: sum(e.output_json.get(k, 0) for e in events if e.action == "llm_call")
                            for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
                        if not any(result.token_usage.values()):
                            result.token_usage = None
        finally:
            factory.kw["bind"].dispose()
    result.retrieved_doc_ids = list(dict.fromkeys(d["doc_id"] for batch in observed.retrievals for d in batch[:3]))
    result.observations["retrievals"] = [[{"doc_id": d["doc_id"], "topic": d["metadata"]["topic"],
                                         "snippet": d["snippet"], "score": d["score"]} for d in batch[:3]] for batch in observed.retrievals]
    if case.safety_label == "unsafe":
        result.unsafe_action_blocked = blocked
    elif case.safety_label == "legal":
        result.false_block = blocked
    if model is not None:
        result.latency_ms = round((perf_counter() - start) * 1000)
    return score(case, result, observed.retrievals)


def run_suite(cases: list[EvalCase], root: Path, index: Any, *, model: Any = None,
              evaluation_class: type[EvaluationService] = EvaluationService) -> dict:
    """One graph implementation, two local dependency toggles, sequential cases only."""
    def execute(case, **kwargs):
        try:
            return run_case(case, root, index, model=model, evaluation_class=evaluation_class, **kwargs)
        except Exception as exc:
            # Preserve failed evaluation rows, never turn infrastructure errors into successful safety blocks.
            return EvalResult(case_id=case.case_id, category=case.category, variant=kwargs.get("variant", "full"),
                recoverable=case.recoverable, safety_label=case.safety_label,
                diagnosis_correct=False if case.expected.error_type else None,
                rag_hit_at_3=False if case.expected.expected_doc_ids or case.expected.expected_doc_topics else None,
                unsafe_action_blocked=False if case.safety_label == "unsafe" else None,
                false_block=True if case.safety_label == "legal" else None,
                recovery_success=False if case.recoverable else None,
                required_tool_recall=0 if case.expected.required_tools else None,
                checks={"execution": False}, notes=f"Execution error: {type(exc).__name__}")
    full = [execute(case) for case in cases]
    rag_cases = [case for case in cases if case.category in {"diagnosis", "rag"}]
    recovery_cases = [case for case in cases if case.recoverable]
    no_rag = [execute(case, rag=False, variant="without_rag") for case in rag_cases]
    diagnosis_only = [execute(case, recovery=False, variant="diagnosis_only") for case in recovery_cases]
    def selected(group):
        return [r for r in full if r.case_id in {c.case_id for c in group}]
    return {"mode": "deepseek" if model is not None else "fake", "embedding_id": index.embedding_id,
        "case_count": len(cases), "results": full, "comparison_results": no_rag + diagnosis_only,
        "overall": calculate_metrics(full),
        "rag_comparison": {"without_rag": calculate_metrics(no_rag), "with_rag": calculate_metrics(selected(rag_cases))},
        "recovery_comparison": {"diagnosis_only": calculate_metrics(diagnosis_only), "full_recovery": calculate_metrics(selected(recovery_cases))},
        "categories": {category: calculate_metrics([r for r in full if r.category == category]) for category in sorted({r.category for r in full})}}
