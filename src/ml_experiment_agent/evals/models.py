from typing import Any, Literal
from pydantic import Field, model_validator
from ..models.experiment import StrictModel
from ..models.enums import ErrorType, Scenario
from ..models.recovery import JsonScalar, RecoveryAction

Category = Literal["planning", "validation", "diagnosis", "rag", "recovery", "safety"]


class EvalExpectation(StrictModel):
    final_status: str | None = None
    error_type: ErrorType | None = None
    required_tools: list[str] = Field(default_factory=list)
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_doc_topics: list[str] = Field(default_factory=list)
    expected_text: list[str] = Field(default_factory=list)
    recovery_action: RecoveryAction | None = None
    retry_count: int | None = Field(default=None, ge=0)
    human_intervention: bool | None = None
    unsafe_action_blocked: bool | None = None
    plan_parameters: list[dict[str, JsonScalar]] | None = None
    run_count: int | None = Field(default=None, ge=1)
    comparison_correct: bool | None = None


class EvalCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9_]+$", max_length=80)
    category: Category
    goal: str = Field(min_length=1)
    operation: Literal["workflow", "config", "job", "rag"] = "workflow"
    scenario: Scenario = Scenario.SUCCESS
    scenarios: list[Scenario] = Field(default_factory=list)
    parameters: list[dict[str, JsonScalar]] = Field(default_factory=lambda: [{"learning_rate": .001}], min_length=1, max_length=5)
    base_config: dict[str, JsonScalar] = Field(default_factory=dict)
    job_overrides: dict[str, JsonScalar] = Field(default_factory=dict)
    log_text: str | None = None
    recovery_enabled: bool = False
    retry_scenario: Scenario = Scenario.SUCCESS
    patch: dict[str, JsonScalar] | None = None
    proposal_doc_ids: list[str] | None = None
    recoverable: bool = False
    safety_label: Literal["unsafe", "legal"] | None = None
    human_response: bool | None = None
    query: str | None = Field(default=None, max_length=400)
    real_llm: bool = False
    expected: EvalExpectation

    @model_validator(mode="after")
    def consistent(self):
        if self.operation == "rag" and not self.query:
            raise ValueError("RAG cases require a query")
        if self.scenarios and len(self.scenarios) != len(self.parameters):
            raise ValueError("scenarios must match the input runs")
        if self.recoverable and not self.recovery_enabled:
            raise ValueError("Recoverable comparison cases must enable recovery")
        if self.recoverable and self.expected.final_status != "SUCCEEDED":
            raise ValueError("Recoverable cases must expect successful completion")
        if self.expected.expected_text and not (self.expected.expected_doc_ids or self.expected.expected_doc_topics):
            raise ValueError("Evidence text expectations require document/topic expectations")
        return self


class EvalResult(StrictModel):
    case_id: str
    category: Category
    variant: str = "full"
    success: bool = False
    final_status: str | None = None
    predicted_error_type: ErrorType | None = None
    diagnosis_correct: bool | None = None
    tools_called: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    rag_hit_at_3: bool | None = None
    required_tool_recall: float | None = Field(default=None, ge=0, le=1)
    recovery_action: RecoveryAction | None = None
    retry_count: int = Field(default=0, ge=0)
    recoverable: bool = False
    human_intervention: bool = False
    recovery_success: bool | None = None
    safety_label: Literal["unsafe", "legal"] | None = None
    unsafe_action_blocked: bool | None = None
    false_block: bool | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    token_usage: dict[str, int] | None = None
    observations: dict[str, Any] = Field(default_factory=dict)
    checks: dict[str, bool] = Field(default_factory=dict)
    notes: str | None = None
