"""Real E5 retrieval smoke; separate from offline pytest and fake demos."""
import argparse
from pathlib import Path
from ml_experiment_agent.rag.documentation import DocumentationIndex
from ml_experiment_agent.rag.embeddings import e5_embeddings, MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    import torch
    torch.set_num_threads(4)
    embedding = e5_embeddings(root / "data/hf_cache", local_files_only=args.local_files_only)
    index = DocumentationIndex.build(root / "knowledge", embedding, MODEL_NAME)
    index.save(root / "data/faiss")
    index = DocumentationIndex.load(root / "data/faiss", embedding, MODEL_NAME)
    results = index.retrieve("CUDA out of memory batch size", top_k=3)
    for result in results:
        print(f"{result['score']:.4f} {result['doc_id']} {result['title']}")
    if not any(r["metadata"]["topic"] in {"cuda_oom", "batch_size"} for r in results):
        raise RuntimeError("Real retrieval did not return OOM/batch-size documentation in top 3")
    print(f"REAL_RAG_SMOKE=PASS model={MODEL_NAME} chunks={len(index.chunks)}")


if __name__ == "__main__":
    main()
