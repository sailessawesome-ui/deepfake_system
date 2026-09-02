from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import STORE  # noqa: E402
from app import tables  # noqa: E402

S, N = "S", "N"


def _specs() -> list[dict]:
    return [{
        "TableName": STORE.audit_table,
        "KeySchema": [
            {"AttributeName": "audit_day", "KeyType": "HASH"},
            {"AttributeName": "event_ts", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "audit_day", "AttributeType": S},
            {"AttributeName": "event_ts", "AttributeType": S},
            {"AttributeName": "user_id", "AttributeType": S},
        ],
        "GlobalSecondaryIndexes": [{
            "IndexName": tables.USER_AUDIT_INDEX,
            "KeySchema": [
                {"AttributeName": "user_id", "KeyType": "HASH"},
                {"AttributeName": "event_ts", "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        }],
        "BillingMode": STORE.billing_mode,
        "_ttl": "expires_at",
    }]


def main() -> int:
    status_only = "--status" in sys.argv

    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        print("boto3 is not installed.  pip install boto3")
        return 2

    client = boto3.client("dynamodb", region_name=STORE.region)

    try:
        client.list_tables()
    except NoCredentialsError:
        print("No AWS credentials. Check AWS_ACCESS_KEY_ID and "
              "AWS_SECRET_ACCESS_KEY in the repo's `env` file.")
        return 2
    except ClientError as exc:
        print(f"Cannot reach DynamoDB in {STORE.region}: {exc}")
        return 2

    print(f"Region {STORE.region}, prefix {STORE.prefix!r}\n")

    for logical in ("users", "sessions", "analyses", "login_attempts"):
        name = tables.table_name(logical)
        try:
            d = client.describe_table(TableName=name)["Table"]
            print(f"  in use   {name}  ({d.get('ItemCount', 0)} items)")
        except ClientError:
            print(f"  MISSING  {name}  <- expected to exist already")

    created = []
    for spec in _specs():
        spec = dict(spec)
        ttl_attr = spec.pop("_ttl")
        name = spec["TableName"]
        try:
            d = client.describe_table(TableName=name)["Table"]
            print(f"  exists   {name}  ({d.get('ItemCount', 0)} items)")
            continue
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ResourceNotFoundException":
                print(f"  ERROR    {name}: {exc}")
                continue
        if status_only:
            print(f"  MISSING  {name}")
            continue
        client.create_table(**spec)
        created.append((name, ttl_attr))
        print(f"  creating {name} ...")

    for name, ttl_attr in created:
        client.get_waiter("table_exists").wait(
            TableName=name, WaiterConfig={"Delay": 3, "MaxAttempts": 40})
        print(f"  ready    {name}")
        for attempt in range(5):
            try:
                client.update_time_to_live(
                    TableName=name,
                    TimeToLiveSpecification={"Enabled": True,
                                             "AttributeName": ttl_attr})
                print(f"  ttl on   {name}.{ttl_attr}")
                break
            except ClientError as exc:
                if attempt == 4:
                    print(f"  ttl FAILED {name}: {exc}")
                else:
                    time.sleep(3)

    print(f"\n{len(created)} created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
