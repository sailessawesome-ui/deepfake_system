from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _parse(uri: str) -> tuple[str, str]:
    rest = uri[5:] if uri.startswith("s3://") else uri
    bucket, _, key = rest.partition("/")
    return bucket, key


def _client():
    import boto3  # type: ignore
    region = os.environ.get("AWS_REGION") or os.environ.get(
        "AWS_DEFAULT_REGION")
    return boto3.client("s3", region_name=region)


def ensure_checkpoint() -> dict:
    from config import INFER

    dest = Path(INFER.checkpoint)
    if dest.exists() and dest.stat().st_size > 0:
        return {"fetched": False, "reason": "already on disk",
                "path": str(dest)}

    uri = os.environ.get("DF_MODEL_S3_URI",
                         os.environ.get("DFD_MODEL_S3_URI", "")).strip()
    if not uri:
        return {"fetched": False, "degraded": True,
                "reason": "no checkpoint on disk and DF_MODEL_S3_URI is unset",
                "path": str(dest)}

    bucket, key = _parse(uri)
    if not bucket or not key:
        return {"fetched": False, "degraded": True,
                "reason": f"malformed DF_MODEL_S3_URI: {uri}"}

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        print(f"[model] downloading {uri} -> {dest}")
        _client().download_file(bucket, key, str(tmp))
        tmp.replace(dest)
        mb = dest.stat().st_size / 1e6
        print(f"[model] checkpoint ready ({mb:.1f} MB)")
        return {"fetched": True, "path": str(dest), "size_mb": round(mb, 1)}
    except Exception as exc:
        print(f"[model] S3 fetch FAILED: {type(exc).__name__}: {exc}")
        return {"fetched": False, "degraded": True,
                "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}


def ensure_calibration() -> dict:
    from config import INFER

    cal = Path(INFER.calibration_file)
    if cal.exists():
        return {"fetched": False, "reason": "already on disk"}

    uri = os.environ.get("DF_MODEL_S3_URI",
                         os.environ.get("DFD_MODEL_S3_URI", "")).strip()
    if not uri:
        return {"fetched": False, "reason": "not configured"}

    bucket, key = _parse(uri)
    cal_key = key.rsplit("/", 1)[0] + "/calibration.json"
    try:
        cal.parent.mkdir(parents=True, exist_ok=True)
        _client().download_file(bucket, cal_key, str(cal))
        print(f"[model] calibration ready from s3://{bucket}/{cal_key}")
        return {"fetched": True}
    except Exception as exc:
        print(f"[model] WARNING: no calibration.json fetched "
              f"({type(exc).__name__}). The engine will use default "
              f"threshold/temperature, so borderline calls may differ "
              f"from your local results.")
        return {"fetched": False, "reason": type(exc).__name__}
