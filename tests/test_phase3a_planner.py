import pytest
from langchain_core.runnables import RunnableLambda
from ml_experiment_agent.llm.planning import DeepSeekExperimentPlanner, StructuredExperimentPlan
from ml_experiment_agent.llm.deepseek import create_deepseek_model


class StructuredFake:
    def __init__(self, result):
        self.result = result
        self.schema = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return RunnableLambda(lambda messages: self.result)


def valid_plan():
    return {"runs": [{"name": "rate_a", "parameters": {"learning_rate": 0.001}}],
            "metrics": ["accuracy", "macro_f1", "ece"]}


def test_deepseek_planner_boundary(environment):
    root, registry, _ = environment
    model = StructuredFake(valid_plan())
    plan = DeepSeekExperimentPlanner(model, root / "data/runs").plan("compare", registry.get("toy_classification"))
    assert model.schema is StructuredExperimentPlan
    assert plan.project_name == "toy_classification" and plan.goal == "compare"
    assert plan.runs[0].parameters == {"learning_rate": 0.001}
    assert plan.runs[0].base_config.output_dir == root / "data/runs"


@pytest.mark.parametrize("parameters", [{"learning_rate": -1}, {"batch_size": 0}, {"precision": "fp16"},
    {"unknown": 1}, {"output_dir": "../external"}, {"entrypoint": "evil.py"}, {"learning_rate": float("nan")}])
def test_invalid_llm_parameters_rejected(environment, parameters):
    root, registry, _ = environment
    value = valid_plan()
    value["runs"][0]["parameters"] = parameters
    with pytest.raises(ValueError):
        DeepSeekExperimentPlanner(StructuredFake(value), root / "data").plan("goal", registry.get("toy_classification"))


@pytest.mark.parametrize("case", ["empty", "too_many", "duplicates", "unknown_metric", "missing_metric", "extra"])
def test_invalid_plan_shape(environment, case):
    root, registry, _ = environment
    value = valid_plan()
    if case == "empty":
        value["runs"] = []
    elif case == "too_many":
        value["runs"] *= 6
    elif case == "duplicates":
        value["runs"] *= 2
    elif case == "unknown_metric":
        value["metrics"] = ["invented"]
    elif case == "missing_metric":
        value["metrics"] = ["accuracy"]
    else:
        value["executable"] = "evil.py"
    with pytest.raises(ValueError):
        DeepSeekExperimentPlanner(StructuredFake(value), root / "data").plan("goal", registry.get("toy_classification"))


def test_project_fields_cannot_be_bypassed(environment):
    root, registry, _ = environment
    spec = registry.get("toy_classification")
    spec.modifiable_config_fields = ["batch_size"]
    with pytest.raises(ValueError, match="forbidden"):
        DeepSeekExperimentPlanner(StructuredFake(valid_plan()), root / "data").plan("goal", spec)


def test_deepseek_configuration_only(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        create_deepseek_model()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-unit-test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    model = create_deepseek_model()  # Construction only; no invoke, no network.
    assert model.model_name == "deepseek-chat" and model.max_retries == 0
