from ...db.models import Experiment
from ...models.experiment import ExperimentDefinition
from ..dependencies import WorkflowDependencies
from ..state import ExperimentGraphState
from .common import audit_once, snapshot


def materialize_runs(state: ExperimentGraphState, deps: WorkflowDependencies) -> ExperimentGraphState:
    experiment = deps.repository.get(Experiment, state["experiment_id"])
    plan = ExperimentDefinition.model_validate(experiment.plan_json)
    ids = [deps.experiment_service.materialize_run(experiment.id, run).id for run in plan.runs]
    audit_once(deps, experiment.id, "runs_materialized", outputs={"run_ids": ids})
    return snapshot(deps, state)
