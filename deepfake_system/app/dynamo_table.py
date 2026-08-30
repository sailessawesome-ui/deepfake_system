"""DynamoDB table with the same surface as app/local_table.Table.

Deliberately a drop-in: same constructor signature, same method names,
same return types. app/tables.py picks one or the other from
DFD_DB_BACKEND, and nothing downstream knows which it got.

Two conversions the callers must not have to think about:

- **Decimal.** DynamoDB has no float type; boto3 hands back Decimal and
  refuses to store float. Everything is converted on the way in and back
  on the way out, so a probability is a plain Python float either side.
- **Empty strings.** Dropped on write. They are legal in non-key
  attributes but read back falsy and cause more confusion than they solve.
"""
from __future__ import annotations

import threading
from decimal import Decimal
from pathlib import Path
from typing import Any

_resource: Any = None
_resource_error: str | None = None
_tried = False
_lock = threading.RLock()


def _resource_for(region: str | None):
    global _resource, _resource_error, _tried
    with _lock:
        if _tried:
            return _resource
        _tried = True
        try:
            import boto3  # type: ignore
            _resource = boto3.resource("dynamodb", region_name=region)
        except ImportError:
            _resource_error = "boto3 not installed"
        except Exception as exc:
            _resource_error = f"{type(exc).__name__}: {str(exc)[:140]}"
        return _resource


def resource_error() -> str | None:
    return _resource_error


def clean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return Decimal(str(round(value, 6)))
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()
                if v is not None and v != ""}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def undecimal(value):
    if isinstance(value, Decimal):
        f = float(value)
        return int(f) if f == int(f) else f
    if isinstance(value, dict):
        return {k: undecimal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [undecimal(v) for v in value]
    return value


class Table:
    def __init__(self, table_name: str, key_schema: tuple[str, str | None],
                 region: str | None = None,
                 local_path: Path | str = "./reports/data.json",
                 prefer_dynamo: bool = True):
        self.name = table_name
        self.pk, self.sk = key_schema
        self.region = region
        self.path = Path(local_path)          # unused; kept for API parity
        self.backend = "dynamodb"
        self.error: str | None = None
        self._table: Any = None

        res = _resource_for(region)
        if res is None:
            self.backend = "unavailable"
            self.error = resource_error()
            return
        try:
            t = res.Table(table_name)
            t.load()                          # DescribeTable: creds + existence
            self._table = t
        except Exception as exc:
            self.backend = "unavailable"
            self.error = f"{type(exc).__name__}: {str(exc)[:140]}"

    @property
    def durable(self) -> bool:
        return self._table is not None

    # ------------------------------------------------------------ writes
    def put(self, item: dict, unique: bool = False) -> bool:
        if self._table is None:
            return False
        item = {k: v for k, v in item.items() if v is not None}
        kwargs: dict[str, Any] = {"Item": clean(item)}
        if unique:
            # The only atomic uniqueness primitive DynamoDB offers.
            kwargs["ConditionExpression"] = f"attribute_not_exists({self.pk})"
        try:
            self._table.put_item(**kwargs)
            return True
        except Exception as exc:
            if type(exc).__name__ == "ConditionalCheckFailedException":
                return False
            self.error = f"{type(exc).__name__}: {str(exc)[:140]}"
            return False

    def put_item(self, item: dict, condition: str = "") -> dict:
        ok = self.put(item, unique=bool(condition))
        return {"stored": ok, "error": None if ok else self.error}

    def delete(self, key: dict) -> bool:
        if self._table is None:
            return False
        try:
            self._table.delete_item(Key=clean(key))
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {str(exc)[:140]}"
            return False

    def delete_item(self, key: dict) -> bool:
        return self.delete(key)

    def update(self, key: dict, updates: dict) -> dict | None:
        """Read-modify-write. Simpler than an UpdateExpression and avoids
        the reserved-word minefield (`status`, `role`, `name` are all
        reserved); at this volume the extra read costs nothing."""
        current = self.get(key) or dict(key)
        current.update(updates)
        return current if self.put(current) else None

    # ------------------------------------------------------------- reads
    def get(self, key: dict) -> dict | None:
        if self._table is None:
            return None
        try:
            got = self._table.get_item(Key=clean(key)).get("Item")
            return undecimal(got) if got else None
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {str(exc)[:140]}"
            return None

    def get_item(self, key: dict) -> dict | None:
        return self.get(key)

    # DynamoDB caps a single Scan or Query page at 1 MB *before* Limit is
    # applied, so one call can return far fewer rows than asked for and
    # still not be the end of the table. `dfd_analyses` rows carry a
    # `report_json` blob, which makes a page run out after roughly seven
    # of them. Paginating is not an optimisation here; without it every
    # read silently under-reports.
    _MAX_PAGES = 20

    def _paginate(self, op, limit: int, **kwargs) -> list[dict]:
        out: list[dict] = []
        pages = 0
        while pages < self._MAX_PAGES:
            try:
                resp = op(**kwargs)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {str(exc)[:140]}"
                break
            out += [undecimal(i) for i in resp.get("Items", [])
                    if isinstance(i, dict)]
            key = resp.get("LastEvaluatedKey")
            pages += 1
            if not key or len(out) >= limit:
                break
            kwargs["ExclusiveStartKey"] = key
        return out[:limit]

    def scan(self, limit: int = 50) -> list[dict]:
        if self._table is None:
            return []
        return self._paginate(self._table.scan, max(limit, 1))

    def query(self, **kwargs) -> list[dict]:
        """Query the base table by partition key.

        Pass the partition value as `key_value`, or hand through raw boto3
        kwargs if you need something more specific.
        """
        if self._table is None:
            return []
        value = kwargs.pop("key_value", None)
        limit = int(kwargs.pop("Limit", 50) or 50)
        if value is not None:
            from boto3.dynamodb.conditions import Key  # type: ignore
            kwargs.setdefault("KeyConditionExpression", Key(self.pk).eq(value))
            kwargs.setdefault("ScanIndexForward", False)
        return self._paginate(self._table.query, max(limit, 1), **kwargs)

    def query_index(self, index_name: str, key_name: str, key_value: Any,
                    limit: int = 50) -> list[dict]:
        if self._table is None:
            return []
        from boto3.dynamodb.conditions import Key  # type: ignore
        return self._paginate(
            self._table.query, max(limit, 1),
            IndexName=index_name,
            KeyConditionExpression=Key(key_name).eq(key_value))

    def status(self) -> dict:
        return {"table": self.name, "backend": self.backend,
                "durable": self.durable, "region": self.region,
                "error": self.error}
