import pytest
from langchain_core.runnables import RunnableLambda
from test_graph import graph_env
from test_phase3a_diagnosis import failed_attempt
from ml_experiment_agent.agent.diagnosis import DiagnosisAgent, diagnosis_from_row
from ml_experiment_agent.db.models import Attempt, Diagnosis, Run, AuditEvent
from ml_experiment_agent.llm.fix_proposer import FixProposer
from ml_experiment_agent.models.enums import Scenario
from ml_experiment_agent.models.experiment import RunConfig
from ml_experiment_agent.models.recovery import ConfigPatch, FixProposal, RecoveryAction as A
from ml_experiment_agent.services.recovery_service import RecoveryService
from ml_experiment_agent.testing import FakeDiagnosisModel, FakeFixModel


def proposal(changes, docs=None):
    return FixProposal(reason="test proposal", patch=ConfigPatch(changes=changes),
                       expected_effect="test effect", supporting_doc_ids=docs or [])


def setup_recovery(graph_env, scenario=Scenario.CUDA_OOM, changes=None):
    root, deps = graph_env
    attempt = failed_attempt(deps, root, scenario)
    DiagnosisAgent(FakeDiagnosisModel(), deps.experiment_service, mode="fake").diagnose(attempt.id)
    model = FakeFixModel(changes)
    recovery = RecoveryService(deps.experiment_service, FixProposer(model))
    return deps, attempt, recovery, model


def test_structured_fix_and_persistence(graph_env):
    deps, attempt, recovery, model = setup_recovery(graph_env)
    recovery.propose(attempt.id)
    recovery.propose(attempt.id)
    assert model.calls == 1
    row = recovery.row(attempt.id)
    result = FixProposal.model_validate(row.proposed_patch_json["proposal"])
    assert result.patch.changes == {"batch_size": 16, "gradient_accumulation_steps": 2}
    decision = recovery.validate(attempt.id)
    assert decision.action == A.AUTO_RETRY and "32 -> 32" in decision.reason
    assert row.decision == "AUTO_RETRY"
    assert deps.repository.list(AuditEvent, action="auto_retry_approved")


@pytest.mark.parametrize("changes", [
    {"nonexistent": 1}, {"batch_size": "16"}, {"batch_size": True}, {"batch_size": 1.5},
    {"batch_size": None}, {"batch_size": 0}, {"learning_rate": -1.0},
    {"precision": "bf16"}, {"output_dir": "../outside"}, {"num_workers": 8},
    {"batch_size": 64}, {}, {"batch_size": 32},
])
def test_invalid_or_unsafe_merged_patch_stops(graph_env, changes):
    deps, attempt, recovery, _ = setup_recovery(graph_env)
    assert recovery.assess(attempt.id, proposal(changes)).action == A.STOP
    assert len(deps.repository.list(Attempt)) == 1


@pytest.mark.parametrize("value", [{"nested": 1}, [1, 2], float("nan"), float("inf")])
def test_flat_finite_scalar_schema(value):
    with pytest.raises(ValueError):
        ConfigPatch(changes={"batch_size": value})


def test_forged_doc_id_rejected_before_persistence(graph_env):
    deps, attempt, recovery, _ = setup_recovery(graph_env)
    class BadModel:
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda _: proposal({"batch_size": 16}, ["invented-doc"]))
    proposer = FixProposer(BadModel())
    with pytest.raises(ValueError, match="unobserved"):
        proposer.propose(diagnosis_from_row(recovery.row(attempt.id)), RunConfig.model_validate(attempt.config_json))
    recovery.proposer = proposer
    recovery.propose(attempt.id)
    assert recovery.row(attempt.id).decision == "STOP"
    assert "invented-doc" not in str(recovery.row(attempt.id).proposed_patch_json)
    assert recovery.assess(attempt.id, proposal({"batch_size": 16}, ["invented-doc"])).action == A.STOP


def test_model_cannot_self_authorize(graph_env):
    _, attempt, recovery, _ = setup_recovery(graph_env)
    class Model:
        def with_structured_output(self, schema, **kwargs):
            value = proposal({"learning_rate": .2}).model_dump() | {"safe": True, "requires_approval": False}
            return RunnableLambda(lambda _: value)
    with pytest.raises(ValueError):
        FixProposer(Model()).propose(diagnosis_from_row(recovery.row(attempt.id)), RunConfig.model_validate(attempt.config_json))


@pytest.mark.parametrize("field,value", [("learning_rate", .2), ("epochs", 2), ("hidden_dim", 16), ("seed", 7)])
def test_scientific_changes_require_approval_even_permissive_policy(graph_env, field, value):
    deps, attempt, recovery, _ = setup_recovery(graph_env)
    deps.policy_engine.config.allowed_auto_fix_fields.append(field)
    deps.policy_engine.config.approval_required_fields = []
    assert recovery.assess(attempt.id, proposal({field: value})).action == A.REQUIRE_APPROVAL


def test_sweep_variable_and_auto_allowlist(graph_env):
    deps, attempt, recovery, _ = setup_recovery(graph_env)
    run = deps.repository.get(Run, attempt.run_id)
    run.parameters_json = {"batch_size": 32, "learning_rate": .001}
    deps.repository.commit()
    assert recovery.assess(attempt.id, proposal({"batch_size": 16})).action == A.REQUIRE_APPROVAL
    assert recovery.assess(attempt.id, proposal({"learning_rate": .002})).action == A.REQUIRE_APPROVAL
    deps.policy_engine.config.allowed_auto_fix_fields = []
    assert recovery.assess(attempt.id, proposal({"gradient_accumulation_steps": 2})).action == A.STOP


def test_project_and_job_policy_rechecked(graph_env):
    deps, attempt, recovery, _ = setup_recovery(graph_env)
    assert recovery.assess(attempt.id, proposal({"batch_size": 16}), resources={"gpu_count": 2}).action == A.STOP
    deps.policy_engine.config.approval_required_fields.append("batch_size")
    assert recovery.assess(attempt.id, proposal({"batch_size": 16})).action == A.REQUIRE_APPROVAL


def test_retry_history_and_idempotency(graph_env):
    deps, attempt, recovery, _ = setup_recovery(graph_env)
    original = attempt.config_json.copy()
    recovery.propose(attempt.id)
    recovery.validate(attempt.id)
    retry = recovery.prepare_retry(attempt.id)
    assert retry.attempt_no == 2 and retry.status == "CREATED" and retry.job_id is None
    assert recovery.prepare_retry(attempt.id).id == retry.id
    assert attempt.status == "FAILED" and attempt.config_json == original
    assert retry.config_json["batch_size"] == 16
    assert retry.config_json["output_dir"] != original["output_dir"]
    assert len(deps.repository.list(Attempt)) == 2
    assert len(deps.repository.list(AuditEvent, action="retry_attempt_created")) == 1


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_max_retries_means_initial_plus_limit(graph_env, limit):
    deps, attempt, recovery, _ = setup_recovery(graph_env)
    deps.policy_engine.config.max_retries = limit
    for index in range(limit + 1):
        recovery.propose(attempt.id)
        decision = recovery.validate(attempt.id)
        if index == limit:
            assert decision.action == A.STOP and "exhausted" in decision.reason
            assert recovery.prepare_retry(attempt.id) is None
            break
        assert decision.action == A.AUTO_RETRY
        attempt = recovery.prepare_retry(attempt.id)
        deps.experiment_service.submit_attempt(attempt.id, Scenario.CUDA_OOM)
        for _ in range(3):
            deps.experiment_service.refresh_attempt_status(attempt.id)
        DiagnosisAgent(FakeDiagnosisModel(), deps.experiment_service, mode="fake").diagnose(attempt.id)
    assert len(deps.repository.list(Attempt)) == limit + 1
    assert deps.repository.list(AuditEvent, action="retry_exhausted")


def test_preempted_no_model_and_same_config(graph_env):
    deps, attempt, recovery, model = setup_recovery(graph_env, Scenario.PREEMPTED)
    recovery.propose(attempt.id)
    assert model.calls == 0
    assert recovery.validate(attempt.id).action == A.AUTO_RETRY
    retry = recovery.prepare_retry(attempt.id)
    assert {k: v for k, v in retry.config_json.items() if k != "output_dir"} == {
        k: v for k, v in attempt.config_json.items() if k != "output_dir"}


@pytest.mark.parametrize("scenario", [Scenario.IMPORT_ERROR, Scenario.PATH_ERROR, Scenario.RUNTIME_ERROR])
def test_unrecoverable_error_never_calls_proposer(graph_env, scenario):
    deps, attempt, recovery, model = setup_recovery(graph_env, scenario)
    recovery.propose(attempt.id)
    assert recovery.validate(attempt.id).action == A.STOP and model.calls == 0
    assert recovery.prepare_retry(attempt.id) is None


def test_validation_rechecked_before_retry_creation(graph_env):
    deps, attempt, recovery, _ = setup_recovery(graph_env)
    recovery.propose(attempt.id)
    assert recovery.validate(attempt.id).action == A.AUTO_RETRY
    deps.policy_engine.config.max_retries = 0
    assert recovery.prepare_retry(attempt.id) is None
    assert len(deps.repository.list(Attempt)) == 1


def test_approval_cannot_bypass_hard_limits(graph_env):
    deps, attempt, recovery, _ = setup_recovery(graph_env, changes={"epochs": 2})
    recovery.propose(attempt.id)
    assert recovery.validate(attempt.id).action == A.REQUIRE_APPROVAL
    assert recovery.prepare_retry(attempt.id) is None
    recovery.approve(attempt.id, True)
    assert recovery.prepare_retry(attempt.id, resources={"memory_gb": 100}) is None


def test_prepared_retry_policy_revocation(graph_env):
    deps, attempt, recovery, _ = setup_recovery(graph_env)
    recovery.propose(attempt.id)
    recovery.validate(attempt.id)
    retry = recovery.prepare_retry(attempt.id)
    assert recovery.validate_submission(retry.id).action == A.AUTO_RETRY
    deps.policy_engine.config.allowed_auto_fix_fields = []
    assert recovery.validate_submission(retry.id).action == A.STOP


def test_prepared_retry_cannot_change_scientific_config(graph_env):
    deps, attempt, recovery, _ = setup_recovery(graph_env)
    recovery.propose(attempt.id)
    recovery.validate(attempt.id)
    retry = recovery.prepare_retry(attempt.id)
    retry.config_json = retry.config_json | {"learning_rate": .9}
    deps.repository.commit()
    assert recovery.validate_submission(retry.id).action == A.STOP


def test_changed_approval_policy_cannot_loop_or_auto_retry(graph_env):
    deps, attempt, recovery, _ = setup_recovery(graph_env)
    recovery.propose(attempt.id)
    assert recovery.validate(attempt.id).action == A.AUTO_RETRY
    deps.policy_engine.config.approval_required_fields.append("batch_size")
    assert recovery.prepare_retry(attempt.id) is None
    assert recovery.row(attempt.id).decision == "STOP"
