"""Import external evidence and resume the same graph through diagnosis/recovery."""
import argparse
from pathlib import Path
from _execution import execution
from ml_experiment_agent.db.models import AuditEvent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--memory-unit", choices=["MB", "GB"], required=True, help="Confirmed unit for any automatically exported retry")
    parser.add_argument("--queue", default="734090d!")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with execution(root, "hpc", memory_unit=args.memory_unit, queue=args.queue) as (repo, service, graph):
        identifier = service.backend.import_result(args.result_dir)
        config = {"configurable": {"thread_id": identifier}}
        saved = graph.get_state(config)
        if saved.next == ("wait_for_poll",):
            graph.invoke(None, config, durability="sync")
        saved = graph.get_state(config)
        print("Experiment:", identifier, "Workflow:", saved.values.get("workflow_status"), "Next:", saved.next)
        if saved.interrupts:
            print("Scientific approval required:", saved.interrupts[0].value)
        for event in repo.list(AuditEvent, experiment_id=identifier, action="hpc_bundle_exported"):
            print("Bundle:", event.output_json["bundle"])
        print("Report:", saved.values.get("report_path"))


if __name__ == "__main__":
    main()
