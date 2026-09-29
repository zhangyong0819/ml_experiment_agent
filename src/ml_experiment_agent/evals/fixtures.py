"""Explicit evaluation input fixtures, never oracle-driven model answers or retrieval results."""
from langchain_core.runnables import RunnableLambda
from ..backends.mock import MockBackend
from ..models.job import JobLog
from ..testing import FakeFixModel


class LogFixtureBackend(MockBackend):
    """Use an input log fixture for unknown errors without changing production scenarios."""
    def __init__(self, policy, text: str | None):
        super().__init__(policy)
        self.text = text

    def logs(self, job_id: str, tail: int = 200) -> JobLog:
        result = super().logs(job_id, tail)
        if self.text is not None and self._get(job_id).state.value in {"FAILED", "TIMED_OUT"}:
            return JobLog(job_id=job_id, text="\n".join(self.text.splitlines()[-tail:]) if tail else "")
        return result


class ProposalFixtureModel(FakeFixModel):
    """Optional hostile document references are inputs to the safety boundary."""
    def __init__(self, changes=None, doc_ids=None):
        super().__init__(changes)
        self.doc_ids = doc_ids

    def with_structured_output(self, schema, **kwargs):
        base = super().with_structured_output(schema, **kwargs)
        def invoke(messages):
            result = base.invoke(messages)
            if self.doc_ids is not None:
                result.supporting_doc_ids = self.doc_ids
            return result
        return RunnableLambda(invoke)


class ObservedIndex:
    """Record actual retrieval batches without changing ranking, queries, snippets, or IDs."""
    def __init__(self, index, enabled=True):
        self.index, self.enabled = index, enabled
        self.embedding_id = index.embedding_id
        self.retrievals: list[list[dict]] = []

    def retrieve(self, query, top_k=3):
        documents = self.index.retrieve(query, top_k) if self.enabled else []
        self.retrievals.append(documents)
        return documents
