from typing import Any
from ..agent.diagnosis import diagnosis_from_row
from ..db.models import Attempt, Diagnosis
from ..llm.fix_proposer import FixProposer
from ..models.experiment import RunConfig
from ..models.recovery import ConfigPatch, FixProposal, RecoveryAction, RecoveryDecision
from .experiment_service import ExperimentService


class RecoveryService:
    """Single-instance recovery: DB history is authoritative; the model only proposes."""
    AUTO_FIELDS = {"batch_size", "gradient_accumulation_steps", "num_workers"}

    def __init__(self, service: ExperimentService, proposer: FixProposer):
        self.service, self.proposer = service, proposer
        self.repo = service.repo

    def row(self, attempt_id: str) -> Diagnosis | None:
        rows = self.repo.list(Diagnosis, attempt_id=attempt_id)
        return rows[0] if rows else None

    def audit(self, attempt_id: str, action: str, outputs: dict[str, Any]) -> None:
        _, run, exp, _ = self.service._context(attempt_id)
        existing = self.repo.latest_audit(exp.id, action, run_id=run.id, attempt_id=attempt_id)
        if existing is None or existing.output_json != outputs:
            self.repo.audit(exp.id, action, run_id=run.id, attempt_id=attempt_id, outputs=outputs)
        self.repo.commit()

    def propose(self, attempt_id: str) -> None:
        row = self.row(attempt_id)
        if row is None or row.proposed_patch_json:
            return
        attempt = self.repo.get(Attempt, attempt_id)
        if attempt.error_type == "PREEMPTED" and row.error_type == "PREEMPTED":
            proposal = FixProposal(reason="Observed scheduler preemption; retry unchanged configuration",
                patch=ConfigPatch(changes={}), expected_effect="A fresh scheduling attempt", supporting_doc_ids=[])
        elif attempt.error_type == "CUDA_OOM" and row.error_type == "CUDA_OOM":
            try:
                proposal = self.proposer.propose(diagnosis_from_row(row), RunConfig.model_validate(attempt.config_json))
            except ValueError:
                self.stop(row, "Invalid or unavailable structured fix proposal")
                return
        else:
            self.stop(row, "No controlled config recovery for this observed error and diagnosis")
            return
        row.proposed_patch_json = {"proposal": proposal.model_dump(mode="json")}
        self.audit(attempt_id, "fix_proposed", {"proposal": proposal.model_dump(mode="json")})

    def stop(self, row: Diagnosis, reason: str) -> RecoveryDecision:
        decision = RecoveryDecision(action=RecoveryAction.STOP, reason=reason)
        row.proposed_patch_json = row.proposed_patch_json | {"decision": decision.model_dump(mode="json")}
        row.decision = decision.action.value
        self.audit(row.attempt_id, "fix_rejected", {"reason": reason})
        return decision

    def assess(self, attempt_id: str, proposal: FixProposal, *, resources: dict | None = None,
               prepared_attempt_id: str | None = None) -> RecoveryDecision:
        """Validate the complete merged config, project, policy, semantics and DB retry budget."""
        attempt, run, _, spec = self.service._context(attempt_id)
        row = self.row(attempt_id)

        def stopped(reason: str) -> RecoveryDecision:
            return RecoveryDecision(action=RecoveryAction.STOP, reason=reason)

        if row is None or attempt.status not in {"FAILED", "TIMED_OUT"}:
            return stopped("A diagnosed failed compute attempt is required")
        if attempt.error_type not in {"CUDA_OOM", "PREEMPTED"} or row.error_type != attempt.error_type:
            return stopped("Error type has no controlled recovery rule")
        current = self.repo.latest_attempt(run.id)
        if prepared_attempt_id is not None:
            link = self.repo.latest_audit(run.experiment_id, "retry_attempt_created", run_id=run.id, attempt_id=attempt.id)
            if not link or link.output_json["retry_attempt_id"] != prepared_attempt_id or current.id != prepared_attempt_id or current.status != "CREATED":
                return stopped("Prepared retry does not match the durable recovery link")
        elif current.id != attempt.id:
            return stopped("Attempt is no longer the current failure")
        # Initial attempt + max_retries; no second retry counter in graph state.
        prospective_total = len(self.repo.list(Attempt, run_id=run.id)) + (0 if prepared_attempt_id else 1)
        if prospective_total > self.service.policy.config.max_retries + 1:
            return stopped("Retry budget exhausted")
        if set(proposal.supporting_doc_ids) - set(row.retrieved_doc_ids_json):
            return stopped("Unobserved supporting document IDs")
        changes = proposal.patch.changes
        fields = set(changes)
        if fields - set(RunConfig.model_fields) or "output_dir" in fields:
            return stopped("Unknown or forbidden path/config field")
        if fields - set(spec.modifiable_config_fields):
            return stopped("ProjectSpec forbids modifying these fields")
        if not fields and attempt.error_type != "PREEMPTED":
            return stopped("An unchanged config cannot remedy this failure")
        if attempt.error_type == "PREEMPTED" and fields:
            return stopped("Preemption retries must preserve config")
        old = RunConfig.model_validate(attempt.config_json)
        for field, value in changes.items():
            original = getattr(old, field)
            allowed_types = (float, int) if type(original) is float else (type(original),)
            if type(value) not in allowed_types:
                return stopped(f"Invalid patch value type for {field}")
        if fields and all(getattr(old, k) == v for k, v in changes.items()):
            return stopped("Patch makes no change")
        try:
            merged = self.service.configs.generate(old, changes, spec)
            if prepared_attempt_id:
                prepared_config = RunConfig.model_validate(current.config_json)
                if prepared_config.model_dump(exclude={"output_dir"}) != merged.model_dump(exclude={"output_dir"}):
                    return stopped("Prepared config differs from validated patch")
                merged = prepared_config
            job = self.service.build_job_spec(run.id, config=merged, **(resources or {}))
            policy = self.service.policy.validate(job, retries=attempt.attempt_no, changed_fields=fields)
        except (ValueError, OSError):
            return stopped("Merged configuration or job specification is invalid")
        if policy.violations:
            return stopped("; ".join(policy.violations))
        # Even a permissive YAML cannot turn scientific decisions into operational fixes.
        scientific = fields - self.AUTO_FIELDS
        sweep = fields & set(run.parameters_json)
        requires = policy.requires_approval or bool(scientific or sweep)
        if requires:
            return RecoveryDecision(action=RecoveryAction.REQUIRE_APPROVAL,
                reason="Scientific, sweep, or policy approval-required field: " + ", ".join(sorted(fields)),
                validated_patch=proposal.patch)
        if attempt.error_type == "CUDA_OOM":
            if merged.batch_size > old.batch_size or merged.gradient_accumulation_steps < old.gradient_accumulation_steps:
                return stopped("OOM automatic recovery must not increase batch size or reduce accumulation")
        effective_old = old.batch_size * old.gradient_accumulation_steps
        effective_new = merged.batch_size * merged.gradient_accumulation_steps
        return RecoveryDecision(action=RecoveryAction.AUTO_RETRY, validated_patch=proposal.patch,
            reason=f"Allowed operational retry; effective batch {effective_old} -> {effective_new}. "
                   "Equal effective batch does not guarantee identical optimization.")

    def validate(self, attempt_id: str, *, resources: dict | None = None) -> RecoveryDecision:
        row = self.row(attempt_id)
        if row is None:
            return RecoveryDecision(action=RecoveryAction.STOP, reason="No persisted diagnosis")
        stored = row.proposed_patch_json
        if "decision" in stored:
            decision = RecoveryDecision.model_validate(stored["decision"])
        elif "proposal" not in stored:
            return self.stop(row, "No valid proposal")
        else:
            proposal = FixProposal.model_validate(stored["proposal"])
            decision = self.assess(attempt_id, proposal, resources=resources)
            row.proposed_patch_json = stored | {"decision": decision.model_dump(mode="json")}
            row.decision = decision.action.value
        self.audit(attempt_id, "fix_validated" if decision.action != RecoveryAction.STOP else "fix_rejected",
                   decision.model_dump(mode="json"))
        if "exhausted" in decision.reason:
            self.audit(attempt_id, "retry_exhausted", {"reason": decision.reason})
        if decision.action == RecoveryAction.AUTO_RETRY:
            self.audit(attempt_id, "auto_retry_approved", decision.model_dump(mode="json"))
        elif decision.action == RecoveryAction.REQUIRE_APPROVAL:
            _, run, exp, _ = self.service._context(attempt_id)
            if not self.repo.latest_audit(exp.id, "human_approval_requested", run_id=run.id, attempt_id=attempt_id):
                self.audit(attempt_id, "human_approval_requested", self.approval_payload(attempt_id))
        return decision

    def approval_payload(self, attempt_id: str) -> dict:
        attempt, run, _, _ = self.service._context(attempt_id)
        row = self.row(attempt_id)
        record = row.proposed_patch_json
        patch = record["proposal"]["patch"]["changes"]
        return {"run_id": run.id, "attempt_id": attempt.id, "error_type": row.error_type,
            "reason": record["proposal"]["reason"], "current_config": {k: attempt.config_json[k] for k in patch},
            "proposed_patch": patch, "why_approval_required": record["decision"]["reason"]}

    def approve(self, attempt_id: str, approved: bool) -> None:
        if type(approved) is not bool:
            raise ValueError("Approval response must be a boolean")
        row = self.row(attempt_id)
        if row.proposed_patch_json.get("approval") is not None:
            return
        if row.decision != RecoveryAction.REQUIRE_APPROVAL:
            raise ValueError("Attempt does not require approval")
        _, run, exp, _ = self.service._context(attempt_id)
        request = self.repo.latest_audit(exp.id, "human_approval_requested", run_id=run.id, attempt_id=attempt_id)
        if request is None or request.output_json != self.approval_payload(attempt_id):
            self.stop(row, "Approval request changed; refuse stale authorization")
            return
        row.proposed_patch_json = row.proposed_patch_json | {"approval": approved}
        self.audit(attempt_id, "human_approved" if approved else "human_rejected", {"approved": approved})

    def prepare_retry(self, attempt_id: str, *, resources: dict | None = None) -> Attempt | None:
        """Recheck at point of use. Replays adopt the same child Attempt, never append another."""
        attempt, run, _, spec = self.service._context(attempt_id)
        row = self.row(attempt_id)
        if row is None:
            return None
        link = self.repo.latest_audit(run.experiment_id, "retry_attempt_created", run_id=run.id, attempt_id=attempt.id)
        if link:
            return self.repo.get(Attempt, link.output_json["retry_attempt_id"])
        stored = row.proposed_patch_json
        if row.decision == RecoveryAction.STOP or stored.get("approval") is False or "decision" not in stored:
            return None
        proposal = FixProposal.model_validate(stored["proposal"])
        decision = self.assess(attempt_id, proposal, resources=resources)
        if decision.action == RecoveryAction.STOP:
            self.stop(row, decision.reason)
            return None
        if decision.action == RecoveryAction.REQUIRE_APPROVAL and stored.get("approval") is not True:
            if row.decision == RecoveryAction.AUTO_RETRY:
                self.stop(row, "Policy now requires approval; prior automatic authorization is invalid")
            return None
        merged = self.service.configs.generate(RunConfig.model_validate(attempt.config_json), proposal.patch.changes, spec)
        return self.service.create_attempt(run.id, config=merged, retry_of=attempt.id)

    def validate_submission(self, attempt_id: str, *, resources: dict | None = None) -> RecoveryDecision:
        """Recheck authorization and current policy after a prepared retry checkpoint resumes."""
        attempt, run, _, _ = self.service._context(attempt_id)
        sources = [a for a in self.repo.list(Attempt, run_id=run.id) if a.attempt_no == attempt.attempt_no - 1]
        row = self.row(sources[0].id) if sources else None
        if row is None or row.decision == RecoveryAction.STOP or row.proposed_patch_json.get("approval") is False:
            return RecoveryDecision(action=RecoveryAction.STOP, reason="No authorized recovery source")
        record = row.proposed_patch_json
        proposal = FixProposal.model_validate(record["proposal"])
        result = self.assess(row.attempt_id, proposal, resources=resources, prepared_attempt_id=attempt_id)
        if result.action == RecoveryAction.REQUIRE_APPROVAL:
            request = self.repo.latest_audit(run.experiment_id, "human_approval_requested", run_id=run.id, attempt_id=row.attempt_id)
            if record.get("approval") is not True or not request or request.output_json != self.approval_payload(row.attempt_id):
                return RecoveryDecision(action=RecoveryAction.STOP, reason="Prepared retry has no valid human approval")
        return result
