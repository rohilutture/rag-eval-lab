"""Chunk a document folder and embed it -> an index directory.

    python scripts/build_index.py --src data/raw --out indexes/c200 --chunk-size 200
    python scripts/build_index.py --src data/raw --out indexes/c400 --chunk-size 400 --overlap 80
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.ingest import ingest_dir, save_chunks
from rag.llm import Embedder
from rag.retrieve import DenseRetriever, save_index

p = argparse.ArgumentParser()
p.add_argument("--src", default="data/raw")
p.add_argument("--out", required=True)
p.add_argument("--chunk-size", type=int, default=200, help="words")
p.add_argument("--overlap", type=int, default=40)
p.add_argument("--embedder", default="st:BAAI/bge-small-en-v1.5")
a = p.parse_args()

t0 = time.perf_counter()
chunks = ingest_dir(Path(a.src), a.chunk_size, a.overlap)
print(f"{len(chunks)} chunks from {len({c.doc_id for c in chunks})} docs")
dense = DenseRetriever(chunks, Embedder(a.embedder))
save_index(Path(a.out), chunks, dense)
save_chunks(chunks, Path(a.out) / "chunks.jsonl")  # human-readable copy
print(f"index -> {a.out} in {time.perf_counter() - t0:.0f}s")
