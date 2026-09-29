from fastapi import APIRouter, HTTPException, Request
from .schemas import CreateExperiment, ResumeExperiment
from ..db.models import Experiment, Run, Attempt, Metric, Diagnosis
from ..db.repository import Repository
from ..models.experiment import ExperimentDefinition

router = APIRouter()


def get(repo, model, identifier):
    try:
        return repo.get(model, identifier)
    except KeyError as exc:
        raise HTTPException(404, "Not found") from exc


def run_summary(repo, run):
    attempts = []
    for attempt in sorted(repo.list(Attempt, run_id=run.id), key=lambda a: a.attempt_no):
        attempts.append({"id": attempt.id, "attempt_no": attempt.attempt_no, "status": attempt.status,
            "backend": attempt.backend, "job_id": attempt.job_id, "error_type": attempt.error_type,
            "diagnoses": [{"error_type": row.error_type, "summary": row.summary,
                           "recovery": row.proposed_patch_json, "decision": row.decision}
                          for row in repo.list(Diagnosis, attempt_id=attempt.id)]})
    return {"id": run.id, "name": run.name, "status": run.status,
            "parameters": run.parameters_json, "attempts": attempts}


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "backend": request.app.state.runner.backend_type}


@router.post("/experiments", status_code=202)
def create(body: CreateExperiment, request: Request):
    runner = request.app.state.runner
    with runner.sessions() as session:
        repo = Repository(session)
        deps = runner.dependencies(repo)
        try:
            experiment = deps.experiment_service.create_experiment(ExperimentDefinition(**body.model_dump()))
        except ValueError as exc:
            raise HTTPException(422, "Unknown registered project") from exc
        repo.audit(experiment.id, "api_created")
        repo.commit()
        identifier = experiment.id
    runner.schedule(identifier)
    return {"experiment_id": identifier, "status": "CREATED"}


@router.get("/experiments/{identifier}")
def experiment(identifier: str, request: Request):
    runner = request.app.state.runner
    with runner.context() as (repo, deps, graph):
        exp = get(repo, Experiment, identifier)
        saved = graph.get_state({"configurable": {"thread_id": identifier}})
        status = exp.status
        if saved.values.get("workflow_status") == "RUNNING" and not repo.latest_audit(identifier, "runner_failed"):
            status = "RUNNING"
        return {"id": exp.id, "goal": exp.goal,
                "status": "AWAITING_APPROVAL" if saved.interrupts else status,
                "execution_status": exp.status,
                "approval": saved.interrupts[0].value if saved.interrupts else None,
                "runs": [run_summary(repo, run) for run in repo.list(Run, experiment_id=exp.id)]}


@router.get("/runs/{identifier}")
def run(identifier: str, request: Request):
    with request.app.state.runner.sessions() as session:
        repo = Repository(session)
        return run_summary(repo, get(repo, Run, identifier))


@router.get("/experiments/{identifier}/metrics")
def metrics(identifier: str, request: Request):
    with request.app.state.runner.sessions() as session:
        repo = Repository(session)
        get(repo, Experiment, identifier)
        return [{"run_id": run.id, "attempt_id": a.id, "metric_name": m.metric_name,
                 "metric_value": m.metric_value, "split": m.split}
                for run in repo.list(Run, experiment_id=identifier)
                for a in repo.list(Attempt, run_id=run.id) for m in repo.list(Metric, attempt_id=a.id)]


@router.get("/experiments/{identifier}/report")
def report(identifier: str, request: Request):
    with request.app.state.runner.context() as (repo, deps, graph):
        get(repo, Experiment, identifier)
        saved = graph.get_state({"configurable": {"thread_id": identifier}})
        return {"report": saved.values.get("final_report"), "ready": bool(saved.values.get("final_report"))}


@router.post("/experiments/{identifier}/resume", status_code=202)
def resume(identifier: str, body: ResumeExperiment, request: Request):
    runner = request.app.state.runner
    with runner.condition:
        with runner.context() as (repo, deps, graph):
            get(repo, Experiment, identifier)
            saved = graph.get_state({"configurable": {"thread_id": identifier}})
            if not saved.interrupts:
                raise HTTPException(409, "Experiment is not awaiting approval")
            previous = repo.latest_audit(identifier, "api_resume_requested")
            interrupt_id = saved.interrupts[0].id
            if previous and previous.input_json.get("interrupt_id") == interrupt_id:
                raise HTTPException(409, "Approval decision already queued")
            repo.audit(identifier, "api_resume_requested", inputs={"approved": body.approved, "interrupt_id": interrupt_id})
            repo.commit()
        runner.schedule(identifier)
    return {"experiment_id": identifier, "status": "RESUME_QUEUED"}
