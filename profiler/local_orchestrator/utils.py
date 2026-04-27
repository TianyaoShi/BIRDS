from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(payload: Any, *, length: int | None = None) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = sha256(encoded.encode("utf-8")).hexdigest()
    if length is None:
        return digest
    return digest[:length]


def slugify(text: str, *, max_length: int = 48) -> str:
    lowered = text.lower()
    slug = _SLUG_PATTERN.sub("-", lowered).strip("-")
    if not slug:
        slug = "item"
    if len(slug) <= max_length:
        return slug
    return slug[:max_length].rstrip("-")


def runtime_server_signature(
    *,
    server_signature_key: str,
    gpu_id: int,
    base_port: int,
    metrics_port: int,
) -> str:
    return stable_hash(
        {
            "server_signature_key": server_signature_key,
            "gpu_id": gpu_id,
            "base_port": base_port,
            "metrics_port": metrics_port,
        }
    )
