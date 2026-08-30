"""Load the repo's `env` file into os.environ.

The file is called `env` — no dot — and sits at the repository root, one
level above this package. It holds the AWS credentials and is gitignored.
Nothing was reading it, so every DFD_* setting in it was inert and the app
silently ran on local files.

Imported for its side effect by config.py, before any setting is read.

Values already present in os.environ win. That ordering matters: a real
deployment injects credentials as service variables, and a stale `env`
file accidentally shipped in the image must not override them.
"""
from __future__ import annotations

import os
from pathlib import Path

# Names that must never be echoed to a log or an API response.
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
        # Tolerate `export FOO=bar` and quoted values.
        if key.startswith("export "):
            key = key[7:].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load(start: Path | None = None) -> dict:
    """Read `env` (or `.env`) from the nearest ancestor that has one."""
    here = (start or Path(__file__).resolve().parent)
    for directory in [here, *here.parents]:
        for name in ("env", ".env"):
            candidate = directory / name
            if not candidate.is_file():
                continue
            values = _parse(candidate)
            applied = []
            for key, value in values.items():
                if key in os.environ:      # already set wins
                    continue
                os.environ[key] = value
                applied.append(key)
            return {"path": str(candidate), "found": len(values),
                    "applied": applied}
    return {"path": None, "found": 0, "applied": []}


def summary(info: dict) -> str:
    """A log line that names the keys loaded but never their values."""
    if not info.get("path"):
        return "[env] no env file found"
    safe = [k for k in info["applied"]
            if not any(s in k.upper() for s in _SECRET)]
    hidden = len(info["applied"]) - len(safe)
    tail = f" (+{hidden} secret)" if hidden else ""
    return (f"[env] {info['path']}: applied {len(info['applied'])}/"
            f"{info['found']} -> {', '.join(sorted(safe))}{tail}")
