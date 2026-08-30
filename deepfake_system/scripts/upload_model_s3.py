"""Put the trained model in S3 so the deployed app can fetch it.

    python scripts/upload_model_s3.py

Creates a private, encrypted bucket if one is not there, uploads
`best.pt` and `calibration.json`, verifies the upload by size, and prints
the exact environment block to paste into Railway.

Both files matter. The weights decide what the model computes;
`calibration.json` carries the threshold and temperature that turn a
logit into a verdict. Deploying one without the other gives you the same
model making different calls near the boundary — which looks exactly like
"the hosted version is less accurate".

Run with your own AWS credentials (the repo's `env` file is loaded
automatically).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import INFER, STORE  # noqa: E402

PREFIX = "models/v1"


def main() -> int:
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        print("boto3 is not installed.  pip install boto3")
        return 2

    ckpt = Path(INFER.checkpoint)
    cal = Path(INFER.calibration_file)
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}.")
        print("Train first, or copy best.pt there. Without it the deployed "
              "app runs the classical baseline and will not match your "
              "reported accuracy.")
        return 2

    region = STORE.region
    try:
        ident = boto3.client("sts", region_name=region).get_caller_identity()
    except NoCredentialsError:
        print("No AWS credentials. Check AWS_ACCESS_KEY_ID / "
              "AWS_SECRET_ACCESS_KEY in the repo's `env` file.")
        return 2
    except ClientError as exc:
        print(f"AWS rejected those credentials: {exc}")
        return 2

    account = ident["Account"]
    bucket = f"deepfake-models-{account}"
    s3 = boto3.client("s3", region_name=region)
    print(f"Account {account}, region {region}\nBucket  {bucket}\n")

    # ---------------------------------------------------------- bucket
    try:
        s3.head_bucket(Bucket=bucket)
        print("  bucket exists")
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "403":
            print("  a bucket with that name exists but is not yours.")
            return 2
        if code not in ("404", "NoSuchBucket"):
            print(f"  ERROR {exc}")
            return 2
        kwargs = {"Bucket": bucket}
        # us-east-1 is the one region that rejects LocationConstraint.
        if region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": region}
        s3.create_bucket(**kwargs)
        print("  bucket created")

    # A public model bucket is the classic AWS mistake. Block it explicitly
    # rather than relying on the account default.
    try:
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True, "IgnorePublicAcls": True,
                "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        s3.put_bucket_encryption(
            Bucket=bucket,
            ServerSideEncryptionConfiguration={"Rules": [{
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256"}}]})
        print("  public access blocked, encryption at rest on")
    except ClientError as exc:
        print(f"  WARNING could not harden bucket: {exc}")

    # ---------------------------------------------------------- upload
    uploads = [(ckpt, f"{PREFIX}/best.pt")]
    if cal.exists():
        uploads.append((cal, f"{PREFIX}/calibration.json"))
    else:
        print(f"\n  WARNING: no calibration.json at {cal}.")
        print("  The deployed engine will fall back to default threshold "
              "and temperature, so borderline verdicts may differ from "
              "your local results.")

    for path, key in uploads:
        mb = path.stat().st_size / 1e6
        print(f"\n  uploading {path.name} ({mb:.1f} MB) ...")
        s3.upload_file(str(path), bucket, key)
        # Verify rather than assume: a silently truncated model is the
        # hardest kind of deployment bug to see.
        remote = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
        if remote != path.stat().st_size:
            print(f"  ERROR size mismatch: local {path.stat().st_size}, "
                  f"remote {remote}")
            return 2
        print(f"  ok  s3://{bucket}/{key}  ({remote} bytes verified)")

    # ---------------------------------------------------------- output
    print("\n" + "=" * 70)
    print("Paste into Railway -> Variables -> Raw Editor.")
    print("Use a RESTRICTED IAM user's key here, not your admin key.")
    print("=" * 70)
    print(f"""
AWS_ACCESS_KEY_ID=<restricted key id>
AWS_SECRET_ACCESS_KEY=<restricted secret>
AWS_REGION={region}

DF_MODEL_S3_URI=s3://{bucket}/{PREFIX}/best.pt

DFD_DB_BACKEND=dynamodb
DFD_TABLE_PREFIX={STORE.prefix}
DFD_REQUIRE_LOGIN=1
DFD_COOKIE_SECURE=1
DFD_FORCE_HTTPS=1
DFD_MAX_UPLOAD_MB=200
DF_CONCURRENCY=2
DFD_EXPLAIN_FRAMES=3
""")
    print("Then: Railway -> Settings -> Root Directory -> deepfake_system")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
