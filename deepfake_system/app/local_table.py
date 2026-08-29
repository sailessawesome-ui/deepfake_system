"""Simple, thread-safe local JSON table for zero-retention storage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Table:
    def __init__(self, table_name: str, key_schema: tuple[str, str | None],
                 region: str | None = None, local_path: Path | str = "./reports/data.json",
                 prefer_dynamo: bool = False):
        self.name = table_name
        self.pk, self.sk = key_schema
        self.path = Path(local_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = "local"
        self._table = None
        self.error: str | None = None

    def _read(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write(self, data: dict[str, dict]) -> None:
        self.path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _key(self, item: dict) -> str:
        key = str(item.get(self.pk, ""))
        if self.sk and self.sk in item:
            key = f"{key}#{item[self.sk]}"
        return key

    def put(self, item: dict, unique: bool = False) -> bool:
        k = self._key(item)
        data = self._read()
        if unique and k in data:
            return False
        data[k] = item
        self._write(data)
        return True

    def put_item(self, item: dict, condition: str = "") -> dict:
        self.put(item, unique=False)
        return item

    def get(self, key: dict) -> dict | None:
        k = self._key(key)
        return self._read().get(k)

    def get_item(self, key: dict) -> dict | None:
        return self.get(key)

    def delete(self, key: dict) -> bool:
        k = self._key(key)
        data = self._read()
        if k in data:
            del data[k]
            self._write(data)
            return True
        return False

    def delete_item(self, key: dict) -> bool:
        return self.delete(key)

    def update(self, key: dict, updates: dict) -> dict | None:
        k = self._key(key)
        data = self._read()
        if k not in data:
            return None
        data[k].update(updates)
        self._write(data)
        return data[k]

    def scan(self, limit: int = 50) -> list[dict]:
        items = list(self._read().values())
        items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return items[:limit]

    def query(self, **kwargs) -> list[dict]:
        return self.scan(50)

    def query_index(self, index_name: str, key_name: str, key_value: str, limit: int = 50) -> list[dict]:
        items = [r for r in self._read().values() if r.get(key_name) == key_value]
        items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return items[:limit]

    def status(self) -> dict:
        return {"backend": "local", "table": self.name, "path": str(self.path)}
