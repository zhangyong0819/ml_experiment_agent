from pathlib import Path

MODEL_NAME = "intfloat/multilingual-e5-small"


def e5_embeddings(cache_folder: Path, *, local_files_only: bool = False):
    """Use official LangChain HF embeddings with E5 query/passage prefixes."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=MODEL_NAME, cache_folder=str(cache_folder),
        model_kwargs={"device": "cpu", "local_files_only": local_files_only},
        encode_kwargs={"normalize_embeddings": True, "prompt": "passage: "},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": "query: "})
