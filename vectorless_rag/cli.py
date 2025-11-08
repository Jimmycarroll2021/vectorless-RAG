import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

from . import VectorlessRAG


def _load_documents(docs_path: str) -> List[Dict]:
    p = Path(docs_path)
    if not p.exists():
        raise FileNotFoundError(f"Documents path not found: {docs_path}")

    docs: List[Dict] = []
    if p.is_dir():
        # Load all .txt files in directory
        for txt in sorted(p.glob("*.txt")):
            text = txt.read_text(encoding="utf-8", errors="ignore")
            docs.append({"id": txt.stem, "text": text, "title": txt.name})
    else:
        # JSONL (one JSON document per line) or JSON list
        text = p.read_text(encoding="utf-8", errors="ignore")
        if p.suffix.lower() in {".jsonl", ".jsonl.txt"}:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                docs.append(json.loads(line))
        else:
            data = json.loads(text)
            if isinstance(data, list):
                docs.extend(data)
            else:
                docs.append(data)
    return docs


def cmd_index_and_query(args: argparse.Namespace) -> int:
    docs = _load_documents(args.docs)
    rag = VectorlessRAG()
    rag.index_documents(docs)

    query = args.query
    if not query and args.query_file:
        query = Path(args.query_file).read_text(encoding="utf-8", errors="ignore")
    if not query:
        print("No query provided. Use --query or --query-file.", file=sys.stderr)
        return 2

    results = rag.retrieve(
        query=query,
        top_k=args.top_k,
        method=args.method,
        expand_query=None if args.expand is None else args.expand,
        rerank=None if args.rerank is None else args.rerank,
    )

    for i, r in enumerate(results, 1):
        title = r.get("title") or r.get("id")
        method = r.get("method", "?")
        score = r.get("score", 0.0)
        print(f"{i:02d}. [{method}] score={score:.3f}  {title}")
        if args.show_text:
            print(r.get("text", "").strip())
            print("-")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vectorless-rag", description="Vectorless RAG CLI")
    sub = p.add_subparsers(dest="command", required=True)

    idx = sub.add_parser("index-and-query", help="Index docs and run a single query")
    idx.add_argument("--docs", required=True, help="Path to docs dir (.txt) or JSON/JSONL file")
    idx.add_argument("--query", help="Query string")
    idx.add_argument("--query-file", help="Read query from file")
    idx.add_argument("--top-k", type=int, default=5, help="Number of results to show")
    idx.add_argument("--method", choices=["bm25", "tfidf", "hybrid"], default="bm25")
    idx.add_argument("--expand", dest="expand", action="store_true", help="Enable query expansion")
    idx.add_argument("--no-expand", dest="expand", action="store_false", help="Disable query expansion")
    idx.set_defaults(expand=None)  # use config default if not specified
    idx.add_argument("--rerank", dest="rerank", action="store_true", help="Enable reranking")
    idx.add_argument("--no-rerank", dest="rerank", action="store_false", help="Disable reranking")
    idx.set_defaults(rerank=None)  # use config default if not specified
    idx.add_argument("--show-text", action="store_true", help="Print document text in output")
    idx.set_defaults(func=cmd_index_and_query)

    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

