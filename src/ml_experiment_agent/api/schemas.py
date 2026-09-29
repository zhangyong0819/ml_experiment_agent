from pydantic import Field, StrictBool
from ..models.experiment import StrictModel


class CreateExperiment(StrictModel):
    goal: str = Field(min_length=1, max_length=4000)
    project_name: str = Field(default="toy_classification", min_length=1, max_length=100)


class ResumeExperiment(StrictModel):
    approved: StrictBool
