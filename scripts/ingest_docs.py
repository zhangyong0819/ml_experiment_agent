"""Build the real E5/FAISS index. Run explicitly; may download the embedding model."""
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
    print(f"Embedding={MODEL_NAME}; documents={len({c['metadata']['doc_id'] for c in index.chunks})}; chunks={len(index.chunks)}")
    print(f"Index: {root / 'data/faiss'}")


if __name__ == "__main__":
    main()
