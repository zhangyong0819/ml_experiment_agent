"""Explicit offline fixtures for tests and MODE=fake demos, never real-model substitutes."""
import hashlib
import json
import re
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from .models.diagnosis import ErrorDiagnosis
from .services.experiment_service import classify_error
from .models.enums import ErrorType

FAKE_EMBEDDING_ID = "fixture-token-hash-v1"


class FixtureEmbeddings(Embeddings):
    """Deterministic token hashing for API/storage tests; not a semantic retrieval model."""
    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 128
        for token in re.findall(r"\w+", text.lower()):
            slot = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % len(vector)
            vector[slot] += 1
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class FakeDiagnosisModel:
    """Scripted tool selection and structured output for offline demonstrations."""
    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        observations = {m.name: json.loads(m.content) for m in messages if isinstance(m, ToolMessage)}
        if "read_job_log" not in observations:
            return AIMessage(content="", tool_calls=[{"name": "read_job_log", "args": {"tail": 30}, "id": "read-log"}])
        text = observations["read_job_log"].get("text", "")
        if "OutOfMemoryError" in text and "retrieve_documentation" not in observations:
            return AIMessage(content="", tool_calls=[{"name": "retrieve_documentation",
                "args": {"query": "CUDA out of memory batch size gradient accumulation", "top_k": 3}, "id": "retrieve-docs"}])
        return AIMessage(content="Observed sufficient evidence. Ready for structured diagnosis.")

    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            observations = {m.name: json.loads(m.content) for m in messages if isinstance(m, ToolMessage)}
            initial = json.loads(next(m.content for m in messages if isinstance(m, HumanMessage)))
            text = observations.get("read_job_log", {}).get("text", initial.get("observed_error_message", ""))
            kind = classify_error(text)
            documents = observations.get("retrieve_documentation", {}).get("documents", [])
            diagnosis = ErrorDiagnosis(error_type=kind, summary=f"Fake-mode diagnosis based on observed log signature: {kind.value}.",
                evidence=[f"Observed: {text.splitlines()[-1][:450]}"] if text else [],
                confidence=0.95 if kind != ErrorType.UNKNOWN else 0.1, retriable=False,
                requires_documentation=False, supporting_doc_ids=list(dict.fromkeys(d["doc_id"] for d in documents)))
            return {"parsed": diagnosis, "raw": AIMessage(content="fixture"), "parsing_error": None} if kwargs.get("include_raw") else diagnosis
        return RunnableLambda(respond)


class FakeFixModel:
    """Offline structured-output fixture; never presented as real LLM verification."""
    def __init__(self, changes=None):
        self.changes = changes
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        def invoke(messages):
            self.calls += 1
            data = json.loads(messages[-1].content)
            config = data["config"]
            batch = max(1, config["batch_size"] // 2)
            changes = self.changes if self.changes is not None else {
                "batch_size": batch,
                "gradient_accumulation_steps": max(1, config["batch_size"] * config["gradient_accumulation_steps"] // batch)}
            reason = "Fake fixture: reduce per-step allocation" if self.changes is None else "Fake fixture: caller-specified config change for policy review"
            effect = "Lower per-step memory use" if self.changes is None else "Exercise policy/approval routing; not a demonstrated OOM remedy"
            return schema.model_validate({"reason": reason,
                "patch": {"changes": changes}, "expected_effect": effect,
                "supporting_doc_ids": data["diagnosis"]["supporting_doc_ids"]})
        return RunnableLambda(invoke)
