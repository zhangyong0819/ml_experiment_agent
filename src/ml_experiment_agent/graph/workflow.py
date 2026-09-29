from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from .dependencies import WorkflowDependencies
from .state import ExperimentGraphState
from .routing import route_after_monitor, route_recovery
from .nodes.planning import plan_experiment
from .nodes.materialization import materialize_runs
from .nodes.validation import validate_configs, policy_check
from .nodes.execution import submit_ready_runs
from .nodes.monitoring import monitor_runs, handle_failed_runs
from .nodes.evaluation import prepare_artifacts, discover_artifacts, evaluate_runs
from .nodes.reporting import compare_runs, generate_report
from .nodes.finalization import finalize_experiment
from .nodes.diagnosis import diagnose_failed_runs
from .nodes.recovery import propose_fixes, validate_recovery, prepare_retry, approval_gate


def build_experiment_graph(dependencies: WorkflowDependencies,
                           checkpointer: BaseCheckpointSaver | None = None, *,
                           deferred_polling: bool = False) -> CompiledStateGraph:
    """Compile a sequential graph; loops and branching are visible LangGraph edges."""
    graph = StateGraph(ExperimentGraphState)
    if deferred_polling and checkpointer is None:
        raise ValueError("Deferred polling requires a persistent checkpointer")

    def wait_for_poll(state: ExperimentGraphState) -> ExperimentGraphState:
        # A scheduler resumes this checkpoint after a poll interval. Reset the
        # per-invocation safety budget; it is never a remote job walltime limit.
        return {"monitor_cycle": 0}

    def monitor_route(state):
        route = route_after_monitor(state)
        return "wait" if deferred_polling and route == "monitor" and not state.get("stop_reason") else route

    graph.add_node("wait_for_poll", wait_for_poll)
    graph.add_edge("wait_for_poll", "monitor_runs")

    def bind(function):
        def node(state: ExperimentGraphState, config: RunnableConfig) -> ExperimentGraphState:
            if not state.get("experiment_id"):
                raise ValueError("No experiment state: resume requires the existing thread_id")
            if str(config.get("configurable", {}).get("thread_id", "")) != state["experiment_id"]:
                raise ValueError("thread_id must equal experiment_id")
            if function is plan_experiment:
                return function(state, config, dependencies)
            return function(state, dependencies)
        return node

    functions = [plan_experiment, materialize_runs, validate_configs, policy_check, submit_ready_runs,
                 monitor_runs, handle_failed_runs, diagnose_failed_runs, propose_fixes, validate_recovery, prepare_artifacts, discover_artifacts,
                 evaluate_runs, compare_runs, generate_report, finalize_experiment]
    for function in functions + [prepare_retry, approval_gate]:
        graph.add_node(function.__name__, bind(function))
    graph.add_edge(START, "plan_experiment")
    for left, right in zip(functions, functions[1:]):
        if left not in (monitor_runs, validate_recovery):
            graph.add_edge(left.__name__, right.__name__)
    graph.add_conditional_edges("monitor_runs", monitor_route,
                                {"monitor": "monitor_runs", "wait": "wait_for_poll", "submit": "submit_ready_runs", "finish": "handle_failed_runs"})
    for source in ("validate_recovery", "approval_gate"):
        graph.add_conditional_edges(source, route_recovery,
            {"retry": "prepare_retry", "approval": "approval_gate", "finish": "prepare_artifacts"})
    graph.add_edge("prepare_retry", "submit_ready_runs")
    graph.add_edge("finalize_experiment", END)
    return graph.compile(checkpointer=checkpointer,
        interrupt_before=["wait_for_poll"] if deferred_polling else None).with_config({"recursion_limit": 4096})
