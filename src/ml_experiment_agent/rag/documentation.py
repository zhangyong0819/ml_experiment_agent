import json
from pathlib import Path
from typing import Any
import faiss
import numpy as np
import yaml
from langchain_core.embeddings import Embeddings
from pydantic import Field
from ..models.experiment import StrictModel
from ..models.enums import ErrorType


class DocMetadata(StrictModel):
    doc_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=160)
    source: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    error_type: ErrorType


def read_chunks(directory: Path, size: int = 700, overlap: int = 80) -> list[dict[str, Any]]:
    """Read trusted Markdown front matter, retain short documents and split long bodies."""
    if size < 100 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk size/overlap")
    chunks, seen = [], set()
    for path in sorted(directory.rglob("*.md")):
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Documentation symlink escapes knowledge directory")
        parts = path.read_text(encoding="utf-8").split("---", 2)
        if len(parts) != 3 or parts[0].strip():
            raise ValueError(f"Missing YAML front matter: {path.name}")
        metadata = DocMetadata.model_validate(yaml.safe_load(parts[1]))
        if metadata.doc_id in seen:
            raise ValueError(f"Duplicate doc_id: {metadata.doc_id}")
        seen.add(metadata.doc_id)
        text = parts[2].strip()
        if not text:
            raise ValueError(f"Empty documentation: {metadata.doc_id}")
        for offset in range(0, len(text), size - overlap):
            chunks.append({"text": text[offset:offset + size], "metadata": metadata.model_dump(mode="json"),
                           "chunk_id": f"{metadata.doc_id}:{offset}"})
            if offset + size >= len(text):
                break
    if not chunks:
        raise ValueError("No Markdown documents found")
    return chunks


class DocumentationIndex:
    """FAISS cosine search with a JSON sidecar; no pickle deserialization or provider framework."""
    def __init__(self, embedding: Embeddings, embedding_id: str, chunks: list[dict], index: Any):
        self.embedding, self.embedding_id, self.chunks, self.index = embedding, embedding_id, chunks, index

    @classmethod
    def build(cls, directory: Path, embedding: Embeddings, embedding_id: str) -> "DocumentationIndex":
        chunks = read_chunks(directory)
        vectors = np.asarray(embedding.embed_documents([c["text"] for c in chunks]), dtype="float32")
        if vectors.ndim != 2 or len(vectors) != len(chunks) or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding output")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(embedding, embedding_id, chunks, index)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "documents.json").write_text(json.dumps({"version": 1, "embedding_id": self.embedding_id,
                                                             "chunks": self.chunks}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embedding: Embeddings, embedding_id: str) -> "DocumentationIndex":
        value = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
        if value["version"] != 1 or value["embedding_id"] != embedding_id:
            raise ValueError("Documentation index embedding/version mismatch; rebuild the index")
        for chunk in value["chunks"]:
            DocMetadata.model_validate(chunk["metadata"])
        index = faiss.read_index(str(directory / "index.faiss"))
        if index.ntotal != len(value["chunks"]):
            raise ValueError("FAISS/metadata count mismatch")
        return cls(embedding, embedding_id, value["chunks"], index)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip() or len(query) > 400 or not 1 <= top_k <= 5:
            raise ValueError("Query must contain 1..400 characters; top_k must be 1..5")
        vector = np.asarray([self.embedding.embed_query(query)], dtype="float32")
        if vector.shape != (1, self.index.d) or not np.isfinite(vector).all():
            raise ValueError("Query embedding dimension/value mismatch")
        faiss.normalize_L2(vector)
        scores, ids = self.index.search(vector, min(top_k, self.index.ntotal))
        result = []
        for score, position in zip(scores[0], ids[0]):
            chunk = self.chunks[int(position)]
            metadata = chunk["metadata"].copy()
            result.append({"doc_id": metadata["doc_id"], "title": metadata["title"],
                           "snippet": chunk["text"][:700], "score": float(score),
                           "metadata": metadata, "chunk_id": chunk["chunk_id"]})
        return result
