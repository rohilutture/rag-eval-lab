"""Load documents (HTML / PDF / Markdown / text) and split them into chunks.

Chunking is structure-aware: split on paragraph boundaries first and pack
paragraphs into ~N-word chunks with overlap. Never cut mid-sentence when
avoidable, because a number separated from its label ("Revenue ... $4.2B") is
the classic way RAG over filings goes wrong.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start: int          # char offset in the cleaned document text
    title: str = ""

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".htm", ".html", ".xml"}:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        for t in soup(["script", "style", "ix:header"]):
            t.decompose()
        # Table cells -> " | " so a row of numbers stays on one line with its label.
        for td in soup.find_all(["td", "th"]):
            td.append(" | ")
        for br in soup.find_all(["br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"]):
            br.append("\n")
        text = soup.get_text()
    elif suffix == ".pdf":
        from pypdf import PdfReader
        text = "\n\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return clean(text)


def clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"( ?\| ?)+", " | ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def paragraphs(text: str):
    pos = 0
    for para in re.split(r"\n\s*\n", text):
        start = text.find(para, pos)
        pos = start + len(para)
        para = para.strip()
        if para:
            yield start, para


def chunk_text(doc_id: str, text: str, size: int = 200, overlap: int = 40,
               title: str = "") -> list[Chunk]:
    """Pack paragraphs into chunks of ~size words; carry ~overlap words forward.
    A paragraph longer than size is split on sentence boundaries."""
    units = []
    for start, para in paragraphs(text):
        words = para.split()
        if len(words) <= size:
            units.append((start, para))
            continue
        sent_pos = start
        buf, buf_start = [], start
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if buf and len(" ".join(buf).split()) + len(sent.split()) > size:
                units.append((buf_start, " ".join(buf)))
                buf, buf_start = [], sent_pos
            buf.append(sent)
            sent_pos += len(sent) + 1
        if buf:
            units.append((buf_start, " ".join(buf)))

    chunks, cur, cur_start = [], [], None
    def emit():
        body = "\n".join(u for _, u in cur)
        cid = hashlib.md5(f"{doc_id}:{cur_start}:{body[:50]}".encode()).hexdigest()[:12]
        chunks.append(Chunk(cid, doc_id, body, cur_start, title))

    for start, unit in units:
        n_cur = sum(len(u.split()) for _, u in cur)
        if cur and n_cur + len(unit.split()) > size:
            emit()
            # overlap: keep trailing units totalling <= overlap words
            keep, n = [], 0
            for s, u in reversed(cur):
                if n + len(u.split()) > overlap:
                    break
                keep.insert(0, (s, u))
                n += len(u.split())
            cur = keep
            cur_start = cur[0][0] if cur else start
        if not cur:
            cur_start = start
        cur.append((start, unit))
    if cur:
        emit()
    return chunks


def ingest_dir(src: Path, size=200, overlap=40) -> list[Chunk]:
    out = []
    for p in sorted(Path(src).rglob("*")):
        if p.is_file() and p.suffix.lower() in {".htm", ".html", ".pdf", ".md", ".txt"}:
            out.extend(chunk_text(p.stem, read_document(p), size, overlap, title=p.stem))
    return out


def save_chunks(chunks, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(c.to_json() for c in chunks), encoding="utf-8")


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l]
