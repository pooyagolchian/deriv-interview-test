"""Small JSON / JSONL IO helpers and a stable content hash for reproducibility."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def append_jsonl(path: str | Path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        fh.write("\n")


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(text)


def content_hash(data: Any) -> str:
    """Stable SHA-256 over the canonical JSON form of ``data`` (order-independent)."""
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
