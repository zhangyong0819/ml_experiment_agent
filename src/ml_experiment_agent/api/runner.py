"""One in-process worker. Application DB and graph checkpoints survive restarts."""
from contextlib import contextmanager
from pathlib import Path
from threading import Condition, Thread
import logging
import time
from langgraph.types import Command
from ..db.models import Experiment
from ..db.repository import Repository
from ..db.session import create_session_factory
from ..graph.checkpoint import sqlite_checkpointer
from ..graph.workflow import build_experiment_graph

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Serialize graph invocations; sessions are owned by each worker invocation.

    The small due-time map schedules polling, not compute jobs. All durable facts
    remain in SQLite. No thread sleeps holding a database transaction.
    """
    def __init__(self, root: Path, dependencies, *, backend_type="mock", poll_interval=30):
        self.root, self.dependencies = root.resolve(), dependencies
        self.backend_type, self.poll_interval = backend_type, poll_interval
        self.database = self.root / "data/api_experiments.sqlite"
        self.checkpoints = self.root / "data/api_checkpoints.sqlite"
        self.sessions = create_session_factory(self.database)
        # Initialize WAL/schema before HTTP reader connections and the worker race.
        with sqlite_checkpointer(self.checkpoints, application_db=self.database):
            pass
        self.condition = Condition()
        self.due: dict[str, float] = {}
        self.closed = False
        self.thread = Thread(target=self._work, daemon=True, name="experiment-runner")

    @contextmanager
    def context(self):
        with self.sessions() as session:
            repo = Repository(session)
            deps = self.dependencies(repo)
            with sqlite_checkpointer(self.checkpoints, application_db=self.database) as saver:
                yield repo, deps, build_experiment_graph(deps, saver,
                    deferred_polling=self.backend_type in {"local", "lsf"})

    def start(self):
        with self.sessions() as session:
            repo = Repository(session)
            for exp in repo.list(Experiment):
                if (repo.latest_audit(exp.id, "api_created")
                        and not repo.latest_audit(exp.id, "experiment_finalized")
                        and not repo.latest_audit(exp.id, "runner_failed")):
                    self.due[exp.id] = time.monotonic()
        self.thread.start()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join()
        self.sessions.kw["bind"].dispose()

    def schedule(self, experiment_id: str, delay=0):
        with self.condition:
            self.due[experiment_id] = time.monotonic() + delay
            self.condition.notify_all()

    def _work(self):
        while True:
            with self.condition:
                if self.closed:
                    return
                if not self.due:
                    self.condition.wait()
                    continue
                identifier = min(self.due, key=self.due.get)
                delay = self.due[identifier] - time.monotonic()
                if delay > 0:
                    self.condition.wait(delay)
                    continue
                del self.due[identifier]
            try:
                self.tick(identifier)
            except Exception as exc:
                # Programming/storage faults stop this experiment, not the worker.
                # Deliberately avoid returning provider/SSH exception text to HTTP.
                logger.error("Workflow stopped for %s (%s)", identifier, type(exc).__name__)
                with self.sessions() as session:
                    repo = Repository(session)
                    exp = repo.get(Experiment, identifier)
                    exp.status = "BLOCKED"
                    repo.audit(identifier, "runner_failed", outcome="failed", outputs={"error": type(exc).__name__})
                    repo.commit()

    def tick(self, identifier: str):
        config = {"configurable": {"thread_id": identifier}}
        with self.context() as (repo, deps, graph):
            exp = repo.get(Experiment, identifier)
            saved = graph.get_state(config)
            if saved.interrupts:
                request = repo.latest_audit(identifier, "api_resume_requested")
                if request is None:
                    return
                # Bind approval to this exact interrupt, never to a later one.
                if request.input_json["interrupt_id"] != saved.interrupts[0].id:
                    return
                value = Command(resume=request.input_json["approved"])
            elif saved.next:
                value = None
            elif saved.values:
                return
            else:
                value = {"experiment_id": identifier, "user_goal": exp.goal}
            graph.invoke(value, config, durability="sync")
            saved = graph.get_state(config)
            if saved.next == ("wait_for_poll",):
                self.schedule(identifier, self.poll_interval)
