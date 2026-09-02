from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

_JUMBF_MARKERS = (b"jumb", b"c2pa", b"caup", b"urn:uuid:")
_C2PA_UUID = bytes.fromhex("d8fec3d61b0e483c92975828877ec481")

_GENERATOR_HINTS = (
    "heygen", "synthesia", "d-id", "did.com", "runway", "pika", "kling",
    "sora", "luma", "hailuo", "veo", "wav2lip", "sadtalker", "faceswap",
    "deepfacelab", "roop", "rope", "facefusion", "akool", "reface",
    "stable diffusion", "comfyui", "animatediff",
)

_CAMERA_HINTS = (
    "com.apple.quicktime", "iphone", "ipad", "samsung", "xiaomi", "oppo",
    "vivo", "huawei", "pixel", "gopro", "canon", "nikon", "sony",
)


def _ffprobe(path: str) -> dict:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", "-show_entries",
             "format_tags:stream_tags", path],
            capture_output=True, text=True, timeout=30).stdout
        return json.loads(out) if out else {}
    except Exception:
        return {}


def _scan_boxes(path: str, limit: int = 4 * 1024 * 1024) -> dict[str, Any]:
    found: dict[str, Any] = {"jumbf_box": False, "c2pa_uuid": False, "offset": None}
    try:
        size = Path(path).stat().st_size
        with open(path, "rb") as f:
            head = f.read(min(limit, size))
            tail = b""
            if size > limit:
                f.seek(max(0, size - limit))
                tail = f.read(limit)
    except OSError:
        return found

    for label, blob, base in (("head", head, 0),
                              ("tail", tail, max(0, size - limit))):
        if not blob:
            continue
        if _C2PA_UUID in blob:
            found["c2pa_uuid"] = True
            found["offset"] = base + blob.find(_C2PA_UUID)
        for marker in _JUMBF_MARKERS:
            idx = blob.find(marker)
            if idx != -1:
                found["jumbf_box"] = True
                if found["offset"] is None:
                    found["offset"] = base + idx
    return found


def _parse_c2pa(path: str) -> dict | None:
    try:
        import c2pa
    except ImportError:
        return None
    try:
        reader = c2pa.Reader.from_file(path)  # type: ignore
        manifest = json.loads(reader.json())
    except Exception as exc:
        return {"valid": False, "error": str(exc)[:200]}

    active = manifest.get("manifests", {}).get(
        manifest.get("active_manifest", ""), {})
    signature = active.get("signature_info", {}) or {}
    actions = []
    for assertion in active.get("assertions", []) or []:
        if "action" in assertion.get("label", ""):
            for a in (assertion.get("data", {}) or {}).get("actions", []) or []:
                if a.get("action"):
                    actions.append(a["action"])

    return {
        "valid": True,
        "claim_generator": active.get("claim_generator"),
        "signed_by": signature.get("issuer"),
        "signed_on": signature.get("time"),
        "title": active.get("title"),
        "actions": actions[:12],
        "validation_errors": [
            e.get("code") for e in
            (manifest.get("validation_status", []) or [])][:6],
    }


def inspect(video_path: str, filename: str = "") -> dict:
    probe = _ffprobe(video_path)
    fmt = probe.get("format", {}) or {}
    tags = {k.lower(): str(v) for k, v in (fmt.get("tags", {}) or {}).items()}
    for stream in probe.get("streams", []) or []:
        for k, v in (stream.get("tags", {}) or {}).items():
            tags.setdefault(k.lower(), str(v))

    blob = " ".join(tags.values()).lower() + " " + (filename or "").lower()

    boxes = _scan_boxes(video_path)
    c2pa_manifest = _parse_c2pa(video_path) if (
        boxes["jumbf_box"] or boxes["c2pa_uuid"]) else None

    if c2pa_manifest and c2pa_manifest.get("valid"):
        c2pa_state = "verified"
    elif c2pa_manifest:
        c2pa_state = "invalid"
    elif boxes["c2pa_uuid"] or boxes["jumbf_box"]:
        c2pa_state = "present_unverified"
    else:
        c2pa_state = "absent"

    generators = sorted({g for g in _GENERATOR_HINTS if g in blob})
    camera = sorted({c for c in _CAMERA_HINTS if c in blob})

    creation = tags.get("creation_time") or tags.get("date")
    interesting = {k: v for k, v in tags.items() if k in (
        "encoder", "creation_time", "make", "model", "software",
        "com.apple.quicktime.model", "com.apple.quicktime.software",
        "handler_name", "comment", "artist", "major_brand")}

    notes = []
    if c2pa_state == "verified":
        notes.append("This file carries a valid Content Credentials (C2PA) "
                     "manifest. The chain of edits below is signed and can "
                     "be checked independently.")
    elif c2pa_state == "present_unverified":
        notes.append("A C2PA manifest box is present but could not be "
                     "validated — install the c2pa package to verify the "
                     "signature rather than just detect the box.")
    elif c2pa_state == "invalid":
        notes.append("A C2PA manifest is present but failed validation. "
                     "Either the file was altered after signing, or the "
                     "signer is not trusted.")
    if generators:
        notes.append("The file metadata names a media-generation tool: "
                     + ", ".join(generators) + ". That is a fact about the "
                     "file, not proof about the face — tools get named for "
                     "legitimate edits too.")
    if not tags:
        notes.append("The container carries no metadata at all. Cameras "
                     "write metadata; messengers strip it. This tells you "
                     "the file was re-processed, not that it was faked.")

    return {
        "c2pa": c2pa_state,
        "c2pa_manifest": c2pa_manifest,
        "jumbf_box_found": boxes["jumbf_box"] or boxes["c2pa_uuid"],
        "generator_hints": generators,
        "camera_hints": camera,
        "creation_time": creation,
        "metadata_present": bool(tags),
        "metadata_fields": len(tags),
        "tags": interesting,
        "notes": notes,
    }


def summarise(report: dict) -> str:
    if report["c2pa"] == "verified":
        return "signed provenance"
    if report["c2pa"] in ("present_unverified", "invalid"):
        return "provenance claim, unverified"
    if report["generator_hints"]:
        return "generator named in metadata"
    if report["camera_hints"]:
        return "camera metadata intact"
    if not report["metadata_present"]:
        return "metadata stripped"
    return "no provenance data"
