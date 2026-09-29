from typing import Annotated
from pydantic import Field
from .experiment import StrictModel
from .enums import ErrorType


class ErrorDiagnosis(StrictModel):
    """Validated observation/inference result; retriable is informational, never an action."""
    error_type: ErrorType
    summary: str = Field(min_length=1, max_length=1200)
    evidence: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(max_length=8)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    retriable: bool
    requires_documentation: bool
    supporting_doc_ids: list[Annotated[str, Field(min_length=1, max_length=120)]] = Field(max_length=10)
