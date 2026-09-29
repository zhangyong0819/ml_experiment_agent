"""Export registered runs; never connects or submits to a cluster."""
import argparse
from pathlib import Path
from _execution import execution, definition
from ml_experiment_agent.db.models import Experiment, AuditEvent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", help="Existing experiment in data/hpc_execution.sqlite")
    parser.add_argument("--memory-unit", choices=["MB", "GB"], required=True, help="Operator-confirmed site unit, never inferred")
    parser.add_argument("--queue", default="734090d!")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with execution(root, "hpc", memory_unit=args.memory_unit, queue=args.queue) as (repo, service, graph):
        exp = repo.get(Experiment, args.experiment_id) if args.experiment_id else service.create_experiment(definition(root, False))
        config = {"configurable": {"thread_id": exp.id}}
        saved = graph.get_state(config)
        if not saved.values:
            graph.invoke({"experiment_id": exp.id, "user_goal": exp.goal}, config, durability="sync")
        print("Experiment:", exp.id, "External execution boundary: manual upload / bsub / download")
        for event in repo.list(AuditEvent, experiment_id=exp.id, action="hpc_bundle_exported"):
            print("Bundle:", event.output_json["bundle"])


if __name__ == "__main__":
    main()
