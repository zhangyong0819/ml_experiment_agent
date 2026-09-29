import json
from pathlib import Path
import pytest
from ml_experiment_agent.rag.documentation import DocumentationIndex, read_chunks
from ml_experiment_agent.testing import FixtureEmbeddings, FAKE_EMBEDDING_ID

KNOWLEDGE = Path(__file__).resolve().parents[1] / "knowledge"


@pytest.fixture
def doc_index():
    return DocumentationIndex.build(KNOWLEDGE, FixtureEmbeddings(), FAKE_EMBEDDING_ID)


def test_ingestion_metadata_and_short_docs():
    chunks = read_chunks(KNOWLEDGE)
    ids = {c["metadata"]["doc_id"] for c in chunks}
    assert len(ids) == len(list(KNOWLEDGE.rglob("*.md")))
    assert {"pytorch-cuda_oom", "troubleshooting-path_error", "cluster-submit_jobs"} <= ids
    assert {p.parent.name for p in KNOWLEDGE.rglob("*.md")} == {"lsf", "cluster", "pytorch", "transformers", "project", "troubleshooting"}
    for chunk in chunks:
        assert len(chunk["text"]) <= 700
        assert {"doc_id", "title", "source", "topic", "error_type"} == set(chunk["metadata"])
        assert chunk["chunk_id"].startswith(chunk["metadata"]["doc_id"])
    assert sum(c["metadata"]["doc_id"] == "troubleshooting-path_error" for c in chunks) == 1


def test_faiss_save_load_and_retrieval(doc_index, tmp_path):
    doc_index.save(tmp_path)
    assert (tmp_path / "index.faiss").is_file() and (tmp_path / "documents.json").is_file()
    assert not list(tmp_path.glob("*.pkl"))
    restored = DocumentationIndex.load(tmp_path, FixtureEmbeddings(), FAKE_EMBEDDING_ID)
    for top_k in (1, 3, 5):
        results = restored.retrieve("CUDA out of memory batch size", top_k)
        assert len(results) == top_k
        assert results == doc_index.retrieve("CUDA out of memory batch size", top_k)
        assert results[0]["score"] >= results[-1]["score"]
        for result in results:
            assert result["doc_id"] in {c["metadata"]["doc_id"] for c in restored.chunks}
            assert len(result["snippet"]) <= 700
            json.dumps(result)


@pytest.mark.parametrize("query,k", [("", 3), ("x" * 401, 3), ("hello", 0), ("hello", 6)])
def test_retrieval_bounds(doc_index, query, k):
    with pytest.raises(ValueError):
        doc_index.retrieve(query, k)


def test_wrong_embedding_id_rejected(doc_index, tmp_path):
    doc_index.save(tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        DocumentationIndex.load(tmp_path, FixtureEmbeddings(), "real-e5")


def test_metadata_corruption_rejected(doc_index, tmp_path):
    doc_index.save(tmp_path)
    path = tmp_path / "documents.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["chunks"].pop()
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="count mismatch"):
        DocumentationIndex.load(tmp_path, FixtureEmbeddings(), FAKE_EMBEDDING_ID)


def test_chunking_and_duplicate_doc_id(tmp_path):
    header = "---\ndoc_id: long\ntitle: Long\nsource: local\ntopic: runtime\nerror_type: UNKNOWN\n---\n"
    (tmp_path / "long.md").write_text(header + "a" * 1500)
    chunks = read_chunks(tmp_path)
    assert len(chunks) == 3 and len(chunks[0]["text"]) == 700
    (tmp_path / "duplicate.md").write_text(header + "short")
    with pytest.raises(ValueError, match="Duplicate"):
        read_chunks(tmp_path)
