"""Sequential offline evaluation by default; --real-llm is an explicit small API subset."""
import argparse
import os
from pathlib import Path
from ml_experiment_agent.evals.runner import load_cases, run_suite
from ml_experiment_agent.evals.reporting import export_results, format_metric
from ml_experiment_agent.rag.documentation import DocumentationIndex
from ml_experiment_agent.testing import FixtureEmbeddings, FAKE_EMBEDDING_ID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-llm", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.real_llm and not os.getenv("DEEPSEEK_API_KEY", "").strip():
        print("SKIP: --real-llm requires DEEPSEEK_API_KEY; no API calls or output overwrite.")
        return
    cases = load_cases(root / "evals/cases")
    model = None
    if args.real_llm:
        from ml_experiment_agent.llm.deepseek import create_deepseek_model
        from ml_experiment_agent.rag.embeddings import e5_embeddings, MODEL_NAME
        cases = [c for c in cases if c.real_llm]
        if not 5 <= len(cases) <= 10:
            raise ValueError("Real subset must contain 5..10 representative cases")
        model = create_deepseek_model()
        embedding = e5_embeddings(root / "data/hf_cache", local_files_only=True)
        index = DocumentationIndex.build(root / "knowledge", embedding, MODEL_NAME)
    else:
        index = DocumentationIndex.build(root / "knowledge", FixtureEmbeddings(), FAKE_EMBEDDING_ID)
    print(f"MODE={'deepseek' if args.real_llm else 'fake'} embedding={index.embedding_id} cases={len(cases)}", flush=True)
    report = run_suite(cases, root, index, model=model)
    output = root / "data/evals" / ("real" if args.real_llm else "latest")
    export_results(report, output)
    for name, value in report["overall"].items():
        print(f"{name}: {format_metric(name, value)}")
    print(f"Output={output}")


if __name__ == "__main__":
    main()
