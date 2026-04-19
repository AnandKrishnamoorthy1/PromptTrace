from __future__ import annotations

import inspect
import json
import sqlite3
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List

DEFAULT_DB_PATH = Path("./prompt_trace.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    version_tag TEXT,
    model TEXT,
    prompt TEXT,
    response TEXT,
    latency_ms INTEGER
);
"""


def _resolve_db_path(db_path: str | Path | None) -> Path:
    if db_path is None:
        return DEFAULT_DB_PATH
    return Path(db_path)


def _ensure_db(db_path: str | Path | None = None) -> Path:
    resolved = _resolve_db_path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(resolved) as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
    return resolved


def _serialize(value: Any) -> str:
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except TypeError:
        return repr(value)


def _extract_model(default_model: str | None, kwargs: Dict[str, Any]) -> str:
    if default_model:
        return default_model
    for key in ("model", "model_name", "llm_model"):
        if key in kwargs and kwargs[key] is not None:
            return str(kwargs[key])
    return "unknown"


def _insert_log(
    *,
    db_path: str | Path | None,
    version_tag: str,
    model: str,
    prompt: str,
    response: str,
    latency_ms: int,
) -> None:
    resolved = _ensure_db(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(resolved) as conn:
        conn.execute(
            """
            INSERT INTO logs (timestamp, version_tag, model, prompt, response, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, version_tag, model, prompt, response, latency_ms),
        )
        conn.commit()


def get_logs(db_path: str | Path | None = None) -> List[Dict[str, Any]]:
    resolved = _ensure_db(db_path)
    with sqlite3.connect(resolved) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, timestamp, version_tag, model, prompt, response, latency_ms
            FROM logs
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def trace_prompt(
    _func: Callable[..., Any] | None = None,
    *,
    model: str | None = None,
    version_tag: str = "dev",
    db_path: str | Path | None = None,
) -> Callable[..., Any]:
    """Decorator that traces prompt-like function calls into SQLite.

    Supports both sync and async functions.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            call_prompt = _serialize({"args": args, "kwargs": kwargs})
            selected_model = _extract_model(model, kwargs)
            try:
                result = func(*args, **kwargs)
                response_payload = _serialize(result)
                return result
            except Exception as exc:
                response_payload = _serialize({"error": repr(exc)})
                raise
            finally:
                latency_ms = int((time.perf_counter() - started) * 1000)
                _insert_log(
                    db_path=db_path,
                    version_tag=version_tag,
                    model=selected_model,
                    prompt=call_prompt,
                    response=response_payload,
                    latency_ms=latency_ms,
                )

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            call_prompt = _serialize({"args": args, "kwargs": kwargs})
            selected_model = _extract_model(model, kwargs)
            try:
                result = await func(*args, **kwargs)
                response_payload = _serialize(result)
                return result
            except Exception as exc:
                response_payload = _serialize({"error": repr(exc)})
                raise
            finally:
                latency_ms = int((time.perf_counter() - started) * 1000)
                _insert_log(
                    db_path=db_path,
                    version_tag=version_tag,
                    model=selected_model,
                    prompt=call_prompt,
                    response=response_payload,
                    latency_ms=latency_ms,
                )

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    if _func is not None and callable(_func):
        return decorator(_func)
    return decorator
