import json
from pathlib import Path
from dataclasses import replace
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from test_graph import graph_env, start, execute, stage
from test_phase3a_rag import doc_index
from ml_experiment_agent.agent.tools import diagnosis_tools
from ml_experiment_agent.agent.diagnosis import DiagnosisAgent
from ml_experiment_agent.db.models import Diagnosis, AuditEvent, Attempt, Run, Metric, Artifact
from ml_experiment_agent.models.diagnosis import ErrorDiagnosis
from ml_experiment_agent.models.enums import Scenario, ErrorType
from ml_experiment_agent.models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ml_experiment_agent.testing import FakeDiagnosisModel
from ml_experiment_agent.graph.dependencies import RunExecutionOptions
from ml_experiment_agent.graph.nodes.diagnosis import diagnose_failed_runs
from ml_experiment_agent.graph.workflow import build_experiment_graph
from ml_experiment_agent.graph.checkpoint import sqlite_checkpointer


def failed_attempt(deps, root, scenario=Scenario.CUDA_OOM):
    service = deps.experiment_service
    exp = service.create_experiment(ExperimentDefinition(goal="diagnose", runs=[RunPlan(name="failed",
        base_config=RunConfig(output_dir=root / "data/runs", epochs=1))]))
    run = service.create_runs(exp.id)[0]
    attempt = service.create_attempt(run.id)
    service.submit_attempt(attempt.id, scenario)
    for _ in range(3):
        service.refresh_attempt_status(attempt.id)
    service.collect_logs(attempt.id)
    return attempt


def test_readonly_tools_scope_outputs_and_audit(graph_env, doc_index):
    root, deps = graph_env
    attempt = failed_attempt(deps, root)
    repo = deps.repository
    repo.save_metric(attempt_id=attempt.id, metric_name="accuracy", metric_value=0.5, split="test")
    repo.commit()
    retrieved = set()
    tools = {t.name: t for t in diagnosis_tools(deps.experiment_service, attempt.id, doc_index, retrieved)}
    assert set(tools) == {"read_job_log", "inspect_run", "retrieve_documentation", "query_metrics", "list_artifacts"}
    log = tools["read_job_log"].invoke({"tail": 1})
    assert "OutOfMemoryError" in log["text"] and len(log["text"].splitlines()) == 1
    info = tools["inspect_run"].invoke({})
    assert info["attempt_status"] == "FAILED" and info["error_type"] == "CUDA_OOM"
    assert info["job_status_source"] == "application_db_last_observed"
    assert tools["query_metrics"].invoke({})["metrics"][0]["metric_value"] == 0.5
    assert tools["list_artifacts"].invoke({})["artifacts"][0]["exists"]
    docs = tools["retrieve_documentation"].invoke({"query": "CUDA out of memory", "top_k": 2})
    assert len(docs["documents"]) == 2 and retrieved
    assert len(repo.list(AuditEvent, action="tool_called", attempt_id=attempt.id)) == 5
    assert repo.list(AuditEvent, action="documents_retrieved", attempt_id=attempt.id)
    # LangChain ignores args for an empty schema; closure scoping still cannot change IDs.
    # DiagnosisAgent additionally validates the schema before every invocation.
    assert tools["inspect_run"].invoke({"attempt_id": "another-attempt"})["attempt_id"] == attempt.id
    with pytest.raises(ValueError):
        tools["read_job_log"].invoke({"tail": 201})
    assert len(repo.list(Attempt)) == 1 and attempt.status == "FAILED"


def test_log_length_and_persisted_logs_survive_backend_loss(graph_env):
    root, deps = graph_env
    attempt = failed_attempt(deps, root)
    path = Path(attempt.config_json["output_dir"]) / "job.log"
    path.write_text("x" * 20000 + "\nfinal line\n")
    deps.backend.jobs.clear()
    tool = diagnosis_tools(deps.experiment_service, attempt.id, None, set())[0]
    result = tool.invoke({})
    assert len(result["text"]) <= 8000 and result["truncated"] and "final line" in result["text"]


def test_diagnosis_with_rag_and_idempotent_persistence(graph_env, doc_index):
    root, deps = graph_env
    attempt = failed_attempt(deps, root)
    model = FakeDiagnosisModel()
    agent = DiagnosisAgent(model, deps.experiment_service, doc_index, mode="fake")
    result = agent.diagnose(attempt.id)
    assert result.error_type == ErrorType.CUDA_OOM and result.supporting_doc_ids
    count = model.calls
    assert agent.diagnose(attempt.id) == result and model.calls == count
    rows = deps.repository.list(Diagnosis, attempt_id=attempt.id)
    assert len(rows) == 1 and rows[0].proposed_patch_json == {}
    assert rows[0].evidence_json["confidence"] == 0.95
    actions = {e.action for e in deps.repository.list(AuditEvent, attempt_id=attempt.id)}
    assert {"llm_diagnosis_started", "tool_called", "documents_retrieved", "diagnosis_completed"} <= actions


def test_path_diagnosis_without_rag(graph_env, doc_index):
    root, deps = graph_env
    attempt = failed_attempt(deps, root, Scenario.PATH_ERROR)
    result = DiagnosisAgent(FakeDiagnosisModel(), deps.experiment_service, doc_index, mode="fake").diagnose(attempt.id)
    assert result.error_type == ErrorType.PATH_ERROR and not result.supporting_doc_ids
    assert not deps.repository.list(AuditEvent, action="documents_retrieved")


def test_direct_structured_diagnosis_no_tools(graph_env):
    root, deps = graph_env
    attempt = failed_attempt(deps, root, Scenario.PATH_ERROR)
    class DirectModel(FakeDiagnosisModel):
        def invoke(self, messages):
            return AIMessage(content="Known FileNotFoundError evidence is sufficient.")
    result = DiagnosisAgent(DirectModel(), deps.experiment_service, mode="fake").diagnose(attempt.id)
    assert result.error_type == ErrorType.PATH_ERROR
    assert not deps.repository.list(AuditEvent, action="tool_called")


def test_tool_loop_bound_returns_unknown(graph_env):
    root, deps = graph_env
    attempt = failed_attempt(deps, root)
    class EndlessModel(FakeDiagnosisModel):
        def invoke(self, messages):
            self.calls += 1
            return AIMessage(content="", tool_calls=[{"name": "inspect_run", "args": {}, "id": str(self.calls)}])
    model = EndlessModel()
    result = DiagnosisAgent(model, deps.experiment_service, mode="fake").diagnose(attempt.id)
    assert model.calls == 4 and result.error_type == ErrorType.UNKNOWN
    assert result.confidence == 0 and "tool_round_limit" in result.summary
    assert len(deps.repository.list(AuditEvent, action="tool_called")) == 4
    assert deps.repository.list(Diagnosis)[0].decision == "diagnosis_failed"


def test_hallucinated_doc_id_is_never_persisted(graph_env, doc_index):
    root, deps = graph_env
    attempt = failed_attempt(deps, root)
    class HallucinatingModel(FakeDiagnosisModel):
        def with_structured_output(self, schema, **kwargs):
            base = super().with_structured_output(schema, **kwargs)
            def corrupt(messages):
                value = base.invoke(messages)
                value["parsed"].supporting_doc_ids = ["invented-secret-document-id"]
                return value
            return RunnableLambda(corrupt)
    result = DiagnosisAgent(HallucinatingModel(), deps.experiment_service, doc_index, mode="fake").diagnose(attempt.id)
    assert result.error_type == ErrorType.UNKNOWN and result.supporting_doc_ids == []
    row = deps.repository.list(Diagnosis)[0]
    assert row.retrieved_doc_ids_json == [] and "unretrieved_document_ids" in row.summary
    assert "invented-secret-document-id" not in json.dumps([a.output_json for a in deps.repository.list(AuditEvent)])


def test_unknown_tool_cannot_mutate_anything(graph_env):
    root, deps = graph_env
    attempt = failed_attempt(deps, root)
    class MaliciousModel(FakeDiagnosisModel):
        def invoke(self, messages):
            return AIMessage(content="", tool_calls=[{"name": "apply_patch", "args": {"batch_size": 1}, "id": "bad"}])
    before = attempt.config_json.copy()
    result = DiagnosisAgent(MaliciousModel(), deps.experiment_service, mode="fake").diagnose(attempt.id)
    assert result.error_type == ErrorType.UNKNOWN
    assert attempt.config_json == before and len(deps.repository.list(Attempt)) == 1
    assert all(a.output_json["rejected"] for a in deps.repository.list(AuditEvent, action="tool_called"))


def test_model_failure_redacts_provider_error(graph_env):
    root, deps = graph_env
    attempt = failed_attempt(deps, root)
    class BrokenModel(FakeDiagnosisModel):
        def invoke(self, messages):
            raise OSError("SECRET_NOT_LOGGED api-key=secret")
    result = DiagnosisAgent(BrokenModel(), deps.experiment_service, mode="fake").diagnose(attempt.id)
    assert result.error_type == ErrorType.UNKNOWN
    events = deps.repository.list(AuditEvent)
    assert "SECRET_NOT_LOGGED" not in json.dumps([a.output_json for a in events])


def test_agent_rejects_invalid_tool_arguments(graph_env):
    root, deps = graph_env
    attempt = failed_attempt(deps, root)
    class InvalidArgsModel(FakeDiagnosisModel):
        def invoke(self, messages):
            return AIMessage(content="", tool_calls=[{"name": "inspect_run", "args": {"attempt_id": "foreign"}, "id": "bad"}])
    result = DiagnosisAgent(InvalidArgsModel(), deps.experiment_service, mode="fake").diagnose(attempt.id)
    assert result.error_type == ErrorType.UNKNOWN
    audits = deps.repository.list(AuditEvent, action="tool_called")
    assert len(audits) == 4 and all(a.output_json["reason"] == "invalid_arguments" for a in audits)


def test_parallel_tool_call_count_is_bounded(graph_env):
    root, deps = graph_env
    attempt = failed_attempt(deps, root)
    class TooManyTools(FakeDiagnosisModel):
        def invoke(self, messages):
            return AIMessage(content="", tool_calls=[{"name": "inspect_run", "args": {}, "id": str(i)} for i in range(6)])
    result = DiagnosisAgent(TooManyTools(), deps.experiment_service, mode="fake").diagnose(attempt.id)
    assert result.error_type == ErrorType.UNKNOWN and "tool_calls_per_round_limit" in result.summary
    assert len(deps.repository.list(AuditEvent, action="tool_called")) == 1


def test_invalid_structured_output_is_unknown(graph_env):
    root, deps = graph_env
    attempt = failed_attempt(deps, root)
    class InvalidOutput(FakeDiagnosisModel):
        def invoke(self, messages):
            return AIMessage(content="done")
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda messages: {"parsed": {"confidence": 7}, "raw": None, "parsing_error": None})
    result = DiagnosisAgent(InvalidOutput(), deps.experiment_service, mode="fake").diagnose(attempt.id)
    assert result.error_type == ErrorType.UNKNOWN and "invalid_structured_diagnosis" in result.summary


def test_optional_retrieval_unavailable(graph_env):
    root, deps = graph_env
    attempt = failed_attempt(deps, root)
    result = DiagnosisAgent(FakeDiagnosisModel(), deps.experiment_service, None, mode="fake").diagnose(attempt.id)
    assert result.error_type == ErrorType.CUDA_OOM and not result.supporting_doc_ids


def test_malformed_tool_id_is_unknown(graph_env):
    root, deps = graph_env
    attempt = failed_attempt(deps, root)
    class MissingID(FakeDiagnosisModel):
        def invoke(self, messages):
            return AIMessage(content="", tool_calls=[{"name": "inspect_run", "args": {}, "id": None}])
    result = DiagnosisAgent(MissingID(), deps.experiment_service, mode="fake").diagnose(attempt.id)
    assert result.error_type == ErrorType.UNKNOWN and "invalid_tool_call_metadata" in result.summary


def test_latency_and_token_metadata(graph_env):
    root, deps = graph_env
    attempt = failed_attempt(deps, root, Scenario.PATH_ERROR)
    class UsageModel(FakeDiagnosisModel):
        def invoke(self, messages):
            response = super().invoke(messages)
            response.usage_metadata = {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
            return response
    DiagnosisAgent(UsageModel(), deps.experiment_service, mode="fake").diagnose(attempt.id)
    calls = deps.repository.list(AuditEvent, action="llm_call")
    assert calls[0].output_json["total_tokens"] == 30
    assert all(c.output_json["latency_ms"] >= 0 for c in calls)


def test_graph_mixed_diagnosis_and_report(graph_env, doc_index):
    root, deps = graph_env
    model = FakeDiagnosisModel()
    deps = replace(deps, execution_options={"run_lr_0001": RunExecutionOptions(Scenario.CUDA_OOM)},
                   diagnosis_agent=DiagnosisAgent(model, deps.experiment_service, doc_index, mode="fake"))
    initial, config = start(deps)
    result, _, _ = execute(root, deps, initial, config)
    assert result["workflow_status"] == "COMPLETED_WITH_FAILURES"
    assert len(result["diagnosed_run_ids"]) == 1
    assert "Failure Diagnosis" in result["final_report"] and "CUDA_OOM" in result["final_report"]
    assert "pytorch-cuda_oom" in result["final_report"]
    assert "messages" not in result and "tools" not in result
    calls = model.calls
    repeated, _, _ = execute(root, deps, initial, config)
    assert repeated == result and model.calls == calls
    assert len(deps.repository.list(Diagnosis)) == 1


def test_multiple_failures_sequentially(graph_env, doc_index):
    root, deps = graph_env
    model = FakeDiagnosisModel()
    options = {name: RunExecutionOptions(Scenario.CUDA_OOM) for name in ("run_lr_001", "run_lr_0005", "run_lr_0001")}
    deps = replace(deps, execution_options=options,
                   diagnosis_agent=DiagnosisAgent(model, deps.experiment_service, doc_index, mode="fake"))
    result, _, _ = execute(root, deps)
    assert result["workflow_status"] == "FAILED" and len(result["diagnosed_run_ids"]) == 3
    assert len(deps.repository.list(Diagnosis)) == 3
    events = [e for e in deps.repository.list(AuditEvent) if e.action in {"llm_diagnosis_started", "diagnosis_completed"}]
    assert [e.action for e in events] == ["llm_diagnosis_started", "diagnosis_completed"] * 3


def test_checkpoint_resume_after_diagnosis_commit(graph_env, doc_index, monkeypatch):
    root, deps = graph_env
    model = FakeDiagnosisModel()
    agent = DiagnosisAgent(model, deps.experiment_service, doc_index, mode="fake")
    deps = replace(deps, execution_options={"run_lr_0001": RunExecutionOptions(Scenario.CUDA_OOM)}, diagnosis_agent=agent)
    original = agent.diagnose
    def crash(attempt_id):
        original(attempt_id)
        raise RuntimeError("crash after diagnosis commit before checkpoint")
    monkeypatch.setattr(agent, "diagnose", crash)
    initial, config = start(deps)
    with pytest.raises(RuntimeError, match="after diagnosis commit"):
        execute(root, deps, initial, config)
    assert len(deps.repository.list(Diagnosis)) == 1
    calls = model.calls
    monkeypatch.setattr(agent, "diagnose", original)
    with sqlite_checkpointer(root / "data/checkpoints.sqlite", application_db=root / "data/application.sqlite") as saver:
        graph = build_experiment_graph(deps, saver)
        assert graph.get_state(config).next == ("diagnose_failed_runs",)
        result = graph.invoke(None, config, durability="sync")
    assert result["workflow_status"] == "COMPLETED_WITH_FAILURES"
    assert model.calls == calls and len(deps.repository.list(Diagnosis)) == 1


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_diagnosis_schema(confidence):
    with pytest.raises(ValueError):
        ErrorDiagnosis(error_type="CUDA_OOM", summary="test", evidence=[], confidence=confidence,
                       retriable=False, requires_documentation=False, supporting_doc_ids=[])
