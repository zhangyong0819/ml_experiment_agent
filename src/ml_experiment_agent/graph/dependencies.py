from dataclasses import dataclass, field
from typing import Any, Protocol
from ..models.enums import Scenario
from ..db.repository import Repository
from ..backends.base import ComputeBackend
from ..services.experiment_service import ExperimentService
from ..services.policy_engine import PolicyEngine
from ..services.evaluation_service import EvaluationService
from .planning import ExperimentPlanner
from ..agent.diagnosis import DiagnosisAgent
from ..services.recovery_service import RecoveryService


class ArtifactPreparer(Protocol):
    def prepare(self, attempt_id: str) -> None:
        """Optional producer bridge for simulation; real backends already produce artifacts."""


@dataclass(frozen=True)
class ToyArtifactPreparer:
    service: ExperimentService

    def prepare(self, attempt_id: str) -> None:
        self.service.prepare_attempt_artifacts(attempt_id)


@dataclass(frozen=True)
class RunExecutionOptions:
    scenario: Scenario = Scenario.SUCCESS
    resources: dict[str, Any] = field(default_factory=dict)
    scenario_sequence: tuple[Scenario, ...] = ()

    def scenario_for(self, attempt_no: int) -> Scenario:
        """Repeat the final fixture outcome when the sequence is exhausted."""
        return self.scenario_sequence[min(attempt_no - 1, len(self.scenario_sequence) - 1)] if self.scenario_sequence else self.scenario


@dataclass(frozen=True)
class WorkflowDependencies:
    """One sequential invocation per container/session; nothing here is checkpointed."""
    experiment_service: ExperimentService
    planner: ExperimentPlanner
    artifact_preparer: ArtifactPreparer | None = None
    execution_options: dict[str, RunExecutionOptions] = field(default_factory=dict)
    diagnosis_agent: DiagnosisAgent | None = None
    recovery_service: RecoveryService | None = None
    default_execution_options: RunExecutionOptions = field(default_factory=RunExecutionOptions)

    @property
    def repository(self) -> Repository:
        return self.experiment_service.repo

    @property
    def backend(self) -> ComputeBackend:
        return self.experiment_service.backend

    @property
    def policy_engine(self) -> PolicyEngine:
        return self.experiment_service.policy

    @property
    def evaluation_service(self) -> EvaluationService:
        return self.experiment_service.evaluation

    def options(self, run_name: str) -> RunExecutionOptions:
        return self.execution_options.get(run_name, self.default_execution_options)
