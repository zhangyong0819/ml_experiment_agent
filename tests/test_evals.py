import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import yaml
from test_graph import FixtureEvaluation
from test_phase3a_rag import doc_index
from ml_experiment_agent.evals.models import EvalCase, EvalExpectation, EvalResult
from ml_experiment_agent.evals.metrics import score, calculate_metrics
from ml_experiment_agent.evals.runner import load_cases, run_case, run_suite, comparison_is_correct
from ml_experiment_agent.evals.reporting import export_results
from ml_experiment_agent.rag.documentation import read_chunks

ROOT = Path(__file__).resolve().parents[1]


def cases():
    return {c.case_id: c for c in load_cases(ROOT / "evals/cases")}


def test_eval_case_parsing_and_distribution():
    values = list(cases().values())
    assert len(values) == 37
    assert 5 <= sum(c.real_llm for c in values) <= 10
    assert {c.category for c in values} == {"planning", "validation", "diagnosis", "rag", "recovery", "safety"}
    for case in values:
        assert EvalCase.model_validate_json(case.model_dump_json()) == case


@pytest.mark.parametrize("override", [{"case_id": "../escape"}, {"category": "made_up"},
    {"operation": "rag"}, {"recoverable": True}, {"scenarios": ["SUCCESS", "CUDA_OOM"]}, {"extra": 1}])
def test_invalid_case_schema(override):
    data = {"case_id": "test", "category": "planning", "goal": "test", "expected": {"final_status": "SUCCEEDED"}}
    with pytest.raises(ValueError):
        EvalCase.model_validate(data | override)


def test_duplicate_case_rejected(tmp_path):
    value = cases()["rag_oom"].model_dump(mode="json")
    for name in ("one", "two"):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="unique"):
        load_cases(tmp_path)


def test_eval_result_json_and_bounds():
    result = EvalResult(case_id="example", category="diagnosis", predicted_error_type="CUDA_OOM")
    assert EvalResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValueError):
        EvalResult(case_id="bad", category="recovery", retry_count=-1)


def result(**values):
    return EvalResult(case_id="fixture", category="diagnosis", **values)


def test_seven_core_metric_denominators():
    rows = [result(success=True, diagnosis_correct=True, rag_hit_at_3=True, recoverable=True,
                   recovery_success=True, retry_count=1, safety_label="legal", false_block=False),
            result(success=False, diagnosis_correct=False, rag_hit_at_3=False, recoverable=True,
                   recovery_success=False, retry_count=2, safety_label="legal", false_block=True),
            result(success=True, safety_label="unsafe", unsafe_action_blocked=True),
            result(success=False, safety_label="unsafe", unsafe_action_blocked=False),
            result(success=True)]
    metrics = calculate_metrics(rows)
    assert metrics["Task Success Rate"] == {"value": .6, "numerator": 3, "denominator": 5}
    for name in ("Diagnosis Accuracy", "RAG Hit@3", "Recovery Success Rate", "Unsafe Action Blocking Rate", "False Blocking Rate"):
        assert metrics[name] == {"value": .5, "numerator": 1, "denominator": 2}
    assert metrics["Average Retry Count"] == {"value": 1.5, "numerator": 3, "denominator": 2}


def test_empty_denominators_are_na():
    assert all(m["value"] is None and m["denominator"] == 0 for m in calculate_metrics([]).values())
    metrics = calculate_metrics([result(success=True)])
    assert metrics["Task Success Rate"]["value"] == 1
    assert metrics["Diagnosis Accuracy"]["value"] is None


def test_required_tool_recall_allows_extra_tools():
    case = cases()["diagnose_cuda_oom"]
    case.expected.expected_doc_ids = []
    observed = result(final_status="FAILED", predicted_error_type="CUDA_OOM",
                      tools_called=["inspect_run", "read_job_log", "retrieve_documentation"])
    actual = score(case, observed, [])
    assert actual.success and actual.required_tool_recall == 1
    missing = score(case, result(final_status="FAILED", predicted_error_type="CUDA_OOM", tools_called=["read_job_log"]), [])
    assert not missing.success and missing.required_tool_recall == .5


def test_rag_hit_requires_actual_top_three_and_evidence():
    case = cases()["rag_group_queues"]
    irrelevant = {"doc_id": "wrong", "metadata": {"topic": "unknown"}, "snippet": ""}
    relevant = {"doc_id": "group_lsf_queues", "metadata": {"topic": "cluster_resource"},
                "snippet": "734090d! 944090d! 9654p6000ib!"}
    row = EvalResult(case_id=case.case_id, category="rag", final_status="RETRIEVED", false_block=False)
    assert score(case, row.model_copy(deep=True), [[irrelevant] * 3 + [relevant]]).rag_hit_at_3 is False
    assert score(case, row.model_copy(deep=True), [[irrelevant, relevant]]).success
    incomplete = relevant | {"snippet": "734090d!"}
    missing = score(case, row.model_copy(deep=True), [[incomplete]])
    assert missing.rag_hit_at_3 and not missing.success


def test_rag_topic_alternative():
    case = EvalCase(case_id="topic", category="rag", operation="rag", query="OOM", goal="retrieve",
                    expected=EvalExpectation(expected_doc_topics=["cuda_oom"]))
    row = EvalResult(case_id="topic", category="rag")
    assert score(case, row, [[{"doc_id": "another", "metadata": {"topic": "cuda_oom"}, "snippet": "memory"}]]).success


def test_recovery_excludes_expected_stop_cases():
    c = cases()
    assert not c["import_stop"].recoverable and not c["retry_exhaustion"].recoverable
    assert c["oom_safe_retry"].recoverable
    fail = score(c["oom_safe_retry"], result(final_status="FAILED", predicted_error_type="CUDA_OOM"), [])
    assert not fail.success and not fail.recovery_success
    recovered = score(c["oom_safe_retry"], result(final_status="SUCCEEDED", predicted_error_type="CUDA_OOM",
        recovery_action="AUTO_RETRY", retry_count=1), [])
    assert recovered.success and recovered.recovery_success


def test_single_evaluation_uses_graph(environment, doc_index):
    root, _, _ = environment
    row = run_case(cases()["oom_safe_retry"], root, doc_index, evaluation_class=FixtureEvaluation)
    assert row.success and row.recovery_success
    assert row.observations["attempt_statuses"] == ["FAILED", "SUCCEEDED"]
    assert row.observations["job_submissions"] == 2
    assert "job_retried" in row.observations["audit_actions"]
    assert row.observations["metrics"][-1]["accuracy"] == .8


def test_multiple_evaluation_and_local_comparisons(environment, doc_index):
    root, _, _ = environment
    c = cases()
    report = run_suite([c[name] for name in ("diagnose_cuda_oom", "oom_safe_retry", "gpu_limit", "legal_config")],
                       root, doc_index, evaluation_class=FixtureEvaluation)
    assert len(report["results"]) == 4 and len(report["comparison_results"]) == 2
    assert report["rag_comparison"]["without_rag"]["RAG Hit@3"]["value"] == 0
    assert report["rag_comparison"]["with_rag"]["RAG Hit@3"]["value"] == 1
    assert report["recovery_comparison"]["diagnosis_only"]["Recovery Success Rate"]["value"] == 0
    assert report["recovery_comparison"]["full_recovery"]["Recovery Success Rate"]["value"] == 1
    assert report["overall"]["False Blocking Rate"]["value"] == 0
    assert report["overall"]["Unsafe Action Blocking Rate"]["value"] == 1


def test_public_cluster_metadata_and_retrieval(doc_index):
    chunks = read_chunks(ROOT / "knowledge")
    manual = [c for c in chunks if c["metadata"]["source"] == "university_cluster_manual"]
    assert len({c["metadata"]["doc_id"] for c in manual}) == 9
    submission = [c for c in manual if c["metadata"]["doc_id"] == "cluster-submit_jobs"]
    assert submission and any("bsub" in c["text"] for c in submission)
    assert all(c["metadata"]["topic"] == "cluster_submit_jobs" for c in submission)
    # Query the actual public corpus, without requiring removed private documents.
    retrieved = doc_index.retrieve(submission[0]["text"][:400], 3)
    assert any(d["doc_id"] == "cluster-submit_jobs" for d in retrieved)
    by_chunk = {c["chunk_id"]: c for c in chunks}
    assert all(d["metadata"] == by_chunk[d["chunk_id"]]["metadata"] for d in retrieved)


def test_cluster_query_case_reports_actual_retriever_output(environment, doc_index):
    root, _, _ = environment
    case = cases()["rag_group_queues"]
    actual = doc_index.retrieve(case.query, 3)
    row = run_case(case, root, doc_index)
    assert row.retrieved_doc_ids == list(dict.fromkeys(d["doc_id"] for d in actual))
    assert row.rag_hit_at_3 == any(d["doc_id"] == "group_lsf_queues" for d in actual)
    assert row.false_block is False


def test_full_fixture_suite_repeatable_exports(environment, doc_index, tmp_path):
    root, _, _ = environment
    selected = list(cases().values())
    first = run_suite(selected, root, doc_index, evaluation_class=FixtureEvaluation)
    second = run_suite(selected, root, doc_index, evaluation_class=FixtureEvaluation)
    assert first == second
    for report, directory in ((first, tmp_path / "one"), (second, tmp_path / "two")):
        export_results(report, directory)
    for name in ("results.json", "results.csv", "summary.md"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    data = json.loads((tmp_path / "one/results.json").read_text(encoding="utf-8"))
    assert data["case_count"] == 37 and len(data["results"]) == 37
    with (tmp_path / "one/results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 37 + len(first["comparison_results"])
    summary = (tmp_path / "one/summary.md").read_text(encoding="utf-8")
    for heading in ("Overall", "RAG comparison", "Recovery comparison", "Category breakdown", "Representative failures"):
        assert f"## {heading}" in summary
    assert "fake" in summary and "not evidence of improved reasoning" in summary


def test_real_llm_without_key_skips_without_network():
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/run_evals.py"), "--real-llm"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0 and "SKIP" in completed.stdout
    assert "no API calls" in completed.stdout


def test_oracle_changes_do_not_change_execution(environment, doc_index):
    root, _, _ = environment
    original = cases()["oom_safe_retry"]
    changed = original.model_copy(deep=True)
    changed.expected.final_status = "FAILED"
    a = run_case(original, root, doc_index, evaluation_class=FixtureEvaluation)
    b = run_case(changed, root, doc_index, evaluation_class=FixtureEvaluation)
    assert a.observations == b.observations and a.final_status == b.final_status
    assert a.success and not b.success


def test_runner_error_not_counted_as_successful_safety_block(environment, doc_index, monkeypatch):
    root, _, _ = environment
    import ml_experiment_agent.evals.runner as module
    def broken(*args, **kwargs):
        raise RuntimeError("sensitive provider details must not be exported")
    monkeypatch.setattr(module, "run_case", broken)
    report = module.run_suite([cases()["gpu_limit"]], root, doc_index)
    row = report["results"][0]
    assert not row.success and row.unsafe_action_blocked is False
    assert row.notes == "Execution error: RuntimeError"


def test_metrics_comparison_detects_wrong_numeric_winner(environment, doc_index, monkeypatch):
    root, _, _ = environment
    from ml_experiment_agent.services.comparison_service import ComparisonService
    compare = ComparisonService.compare
    def wrong(self, experiment_id):
        value = compare(self, experiment_id)
        value.metric_results["accuracy"].value = .123
        return value
    monkeypatch.setattr(ComparisonService, "compare", wrong)
    row = run_case(cases()["metrics_comparison"], root, doc_index, evaluation_class=FixtureEvaluation)
    assert not row.success and row.checks["comparison_correct"] is False


def test_duplicate_submission_would_fail_safety_integrity(environment, doc_index, monkeypatch):
    root, _, _ = environment
    from ml_experiment_agent.evals.fixtures import LogFixtureBackend
    submit = LogFixtureBackend.submit
    def duplicate(self, job):
        handle = submit(self, job)
        # Simulate an externally duplicated job record without another persisted Attempt.
        self.jobs["duplicate"] = self.jobs[handle.job_id]
        return handle
    monkeypatch.setattr(LogFixtureBackend, "submit", duplicate)
    row = run_case(cases()["protected_learning_rate"], root, doc_index, evaluation_class=FixtureEvaluation)
    assert not row.success and row.unsafe_action_blocked is False
    assert row.checks["submission_integrity"] is False
