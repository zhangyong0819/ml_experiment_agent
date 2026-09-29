"""Explicit opt-in real API smoke. Not collected by pytest (testpaths=tests)."""
from pydantic import BaseModel
from langchain_core.tools import tool
from ml_experiment_agent.llm.deepseek import create_deepseek_model


class SmokeResult(BaseModel):
    message: str


@tool
def smoke_observation() -> str:
    """Return the fixed local smoke-test observation, without external side effects."""
    return "ok"


def main() -> None:
    model = create_deepseek_model()
    result = model.with_structured_output(SmokeResult, method="function_calling").invoke("Return message='ok'.")
    validated = SmokeResult.model_validate(result)
    response = model.bind_tools([smoke_observation], tool_choice="smoke_observation").invoke("Call smoke_observation once.")
    if len(response.tool_calls) != 1 or response.tool_calls[0]["name"] != "smoke_observation":
        raise RuntimeError("DeepSeek did not return the requested tool call")
    assert smoke_observation.invoke(response.tool_calls[0]["args"]) == "ok"
    print(f"REAL_DEEPSEEK_SMOKE=PASS structured_message={validated.message}")


if __name__ == "__main__":
    main()
