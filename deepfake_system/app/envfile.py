from __future__ import annotations

import os
from pathlib import Path

_SECRET = ("SECRET", "PASSWORD", "TOKEN", "CREDENTIAL")


def _parse(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf8")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load(start: Path | None = None) -> dict:
    here = (start or Path(__file__).resolve().parent)
    for directory in [here, *here.parents]:
        for name in ("env", ".env"):
            candidate = directory / name
            if not candidate.is_file():
                continue
            values = _parse(candidate)
            applied = []
            for key, value in values.items():
                if key in os.environ:     
                    continue
                os.environ[key] = value
                applied.append(key)
            return {"path": str(candidate), "found": len(values),
                    "applied": applied}
    return {"path": None, "found": 0, "applied": []}


def summary(info: dict) -> str:
    if not info.get("path"):
        return "[env] no env file found"
    safe = [k for k in info["applied"]
            if not any(s in k.upper() for s in _SECRET)]
    hidden = len(info["applied"]) - len(safe)
    tail = f" (+{hidden} secret)" if hidden else ""
    return (f"[env] {info['path']}: applied {len(info['applied'])}/"
            f"{info['found']} -> {', '.join(sorted(safe))}{tail}")
