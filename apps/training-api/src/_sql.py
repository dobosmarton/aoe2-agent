"""Typed thin wrappers over sqlite3.

The sqlite3 driver returns `Any` from `fetchone`/`fetchall` and `Row.__getitem__`,
and `json.loads` returns `Any`. This module is the single boundary where those
library-imposed `Any` values are cast to concrete types — the only sanctioned use
of `cast` under the project's no-`Any` rule. Everything above this layer is typed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypeAlias, cast

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping, Sequence

SqlParams: TypeAlias = "Sequence[object] | Mapping[str, object]"


def query_one(conn: sqlite3.Connection, sql: str, params: SqlParams = ()) -> sqlite3.Row | None:
    return cast("sqlite3.Row | None", conn.execute(sql, params).fetchone())


def query_all(conn: sqlite3.Connection, sql: str, params: SqlParams = ()) -> list[sqlite3.Row]:
    return cast("list[sqlite3.Row]", conn.execute(sql, params).fetchall())


def col_int(row: sqlite3.Row, key: str) -> int:
    return cast("int", row[key])


def col_str(row: sqlite3.Row, key: str) -> str:
    return cast("str", row[key])


def col_opt_str(row: sqlite3.Row, key: str) -> str | None:
    return cast("str | None", row[key])


def loads_object(raw: str) -> object:
    """`json.loads` typed as `object` — forces isinstance narrowing at call sites."""
    return cast("object", json.loads(raw))


def loads_dict(raw: str) -> dict[str, object] | None:
    """Parse JSON expected to be an object; `None` if it is any other shape.

    Values are typed `object` (not `Any`), so call sites must narrow them.
    """
    parsed = cast("object", json.loads(raw))
    if isinstance(parsed, dict):
        return cast("dict[str, object]", parsed)
    return None
