from enum import StrEnum
from typing import Annotated
from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr
from .experiment import StrictModel

JsonScalar = StrictBool | StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)] | StrictStr | None


class ConfigPatch(StrictModel):
    """Only flat RunConfig field assignments, never files or scheduler commands."""
    changes: dict[str, JsonScalar] = Field(max_length=12)


class FixProposal(StrictModel):
    reason: str = Field(min_length=1, max_length=1200)
    patch: ConfigPatch
    expected_effect: str = Field(min_length=1, max_length=1200)
    supporting_doc_ids: list[Annotated[str, Field(min_length=1, max_length=120)]] = Field(default_factory=list, max_length=10)


class RecoveryAction(StrEnum):
    AUTO_RETRY = "AUTO_RETRY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    STOP = "STOP"


class RecoveryDecision(StrictModel):
    action: RecoveryAction
    reason: str
    validated_patch: ConfigPatch | None = None
