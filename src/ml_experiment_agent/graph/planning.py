from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from ..models.experiment import ExperimentDefinition, RunPlan, RunConfig
from ..models.project import ProjectSpec


class ExperimentPlanner(Protocol):
    def plan(self, goal: str, project_spec: ProjectSpec) -> ExperimentDefinition:
        """Return a structured candidate plan; numerical comparison belongs to services."""


@dataclass(frozen=True)
class DeterministicPlanner:
    output_dir: Path
    epochs: int = 3

    def plan(self, goal: str, project_spec: ProjectSpec) -> ExperimentDefinition:
        return ExperimentDefinition(goal=goal, project_name=project_spec.project_name, runs=[
            RunPlan(name=name, parameters={"learning_rate": rate},
                    base_config=RunConfig(output_dir=self.output_dir, epochs=self.epochs))
            for name, rate in (("run_lr_001", 0.001), ("run_lr_0005", 0.0005), ("run_lr_0001", 0.0001))])


@dataclass(frozen=True)
class FixturePlanner:
    definition: ExperimentDefinition

    def plan(self, goal: str, project_spec: ProjectSpec) -> ExperimentDefinition:
        result = self.definition.model_copy(deep=True)
        result.goal = goal
        if result.project_name != project_spec.project_name:
            raise ValueError("Fixture project does not match registered experiment")
        return result
