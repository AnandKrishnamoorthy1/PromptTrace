from __future__ import annotations

import inspect
import json
import sqlite3
import time
from contextlib import closing, contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List
from uuid import uuid4

try:
    import tiktoken
except ImportError:
    tiktoken = None

DEFAULT_DB_PATH = Path("./prompt_trace.db")

_active_run_id: ContextVar[str | None] = ContextVar("_active_run_id", default=None)
_active_trace_id: ContextVar[str | None] = ContextVar("_active_trace_id", default=None)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    version_tag TEXT,
    model TEXT,
    prompt TEXT,
    response TEXT,
    latency_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    run_id TEXT,
    trace_id TEXT,
    parent_trace_id TEXT,
    agent_name TEXT,
    step_name TEXT
);
"""

TOKEN_COLUMNS = {
    "prompt_tokens": "INTEGER",
    "completion_tokens": "INTEGER",
    "total_tokens": "INTEGER",
}

TRACE_COLUMNS = {
    "run_id": "TEXT",
    "trace_id": "TEXT",
    "parent_trace_id": "TEXT",
    "agent_name": "TEXT",
    "step_name": "TEXT",
}


def _resolve_db_path(db_path: str | Path | None) -> Path:
    if db_path is None:
        return DEFAULT_DB_PATH
    return Path(db_path)


def _ensure_db(db_path: str | Path | None = None) -> Path:
    resolved = _resolve_db_path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(resolved)) as conn:
        conn.execute(CREATE_TABLE_SQL)
        existing_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(logs)").fetchall()
        }
        for col_name, col_type in TOKEN_COLUMNS.items():
            if col_name not in existing_columns:
                conn.execute(f"ALTER TABLE logs ADD COLUMN {col_name} {col_type}")
        for col_name, col_type in TRACE_COLUMNS.items():
            if col_name not in existing_columns:
                conn.execute(f"ALTER TABLE logs ADD COLUMN {col_name} {col_type}")
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


def _extract_provider_usage(result: Any) -> Dict[str, int] | None:
    usage_obj = None
    if isinstance(result, dict):
        usage_obj = result.get("usage")
    else:
        usage_obj = getattr(result, "usage", None)

    if usage_obj is None:
        return None

    if not isinstance(usage_obj, dict):
        usage_obj = {
            key: getattr(usage_obj, key, None)
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
            )
        }

    prompt_tokens = usage_obj.get("prompt_tokens")
    completion_tokens = usage_obj.get("completion_tokens")
    total_tokens = usage_obj.get("total_tokens")

    # Common alt names used by several SDKs.
    if prompt_tokens is None:
        prompt_tokens = usage_obj.get("input_tokens")
    if completion_tokens is None:
        completion_tokens = usage_obj.get("output_tokens")

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    prompt_int = int(prompt_tokens) if prompt_tokens is not None else 0
    completion_int = int(completion_tokens) if completion_tokens is not None else 0
    total_int = int(total_tokens) if total_tokens is not None else (prompt_int + completion_int)

    return {
        "prompt_tokens": prompt_int,
        "completion_tokens": completion_int,
        "total_tokens": total_int,
    }


def _estimate_tokens(text: str, model_name: str) -> int:
    if not text:
        return 0

    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    # Fallback when tiktoken is not installed.
    approx = int(len(text.split()) * 1.3)
    return max(1, approx)


def _resolve_token_usage(
    *,
    result: Any,
    prompt_text: str,
    response_text: str,
    model_name: str,
    usage_extractor: Callable[[Any], Dict[str, int] | None] | None,
) -> Dict[str, int]:
    if usage_extractor is not None:
        custom = usage_extractor(result)
        if custom is not None:
            return {
                "prompt_tokens": int(custom.get("prompt_tokens", 0)),
                "completion_tokens": int(custom.get("completion_tokens", 0)),
                "total_tokens": int(
                    custom.get(
                        "total_tokens",
                        int(custom.get("prompt_tokens", 0)) + int(custom.get("completion_tokens", 0)),
                    )
                ),
            }

    provider_usage = _extract_provider_usage(result)
    if provider_usage is not None:
        return provider_usage

    prompt_tokens = _estimate_tokens(prompt_text, model_name)
    completion_tokens = _estimate_tokens(response_text, model_name)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _extract_prompt_value(
    *,
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
    prompt_arg_name: str,
    prompt_extractor: Callable[[tuple[Any, ...], Dict[str, Any], Dict[str, Any]], Any] | None,
) -> Any:
    bound_args: Dict[str, Any] = {}
    try:
        bound_args = dict(signature.bind_partial(*args, **kwargs).arguments)
    except TypeError:
        bound_args = {}

    if prompt_extractor is not None:
        return prompt_extractor(args, kwargs, bound_args)

    if prompt_arg_name in bound_args:
        return bound_args[prompt_arg_name]
    if prompt_arg_name in kwargs:
        return kwargs[prompt_arg_name]
    if args:
        return args[0]
    return {"args": args, "kwargs": kwargs}


def _insert_log(
    *,
    db_path: str | Path | None,
    version_tag: str,
    model: str,
    prompt: str,
    response: str,
    latency_ms: int,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    run_id: str | None = None,
    trace_id: str | None = None,
    parent_trace_id: str | None = None,
    agent_name: str | None = None,
    step_name: str | None = None,
) -> None:
    resolved = _ensure_db(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(resolved)) as conn:
        conn.execute(
            """
            INSERT INTO logs (
                timestamp,
                version_tag,
                model,
                prompt,
                response,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                run_id,
                trace_id,
                parent_trace_id,
                agent_name,
                step_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                version_tag,
                model,
                prompt,
                response,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                run_id,
                trace_id,
                parent_trace_id,
                agent_name,
                step_name,
            ),
        )
        conn.commit()


@contextmanager
def trace_run(run_id: str | None = None):
    """Create an explicit run boundary for nested traced calls."""
    resolved_run_id = run_id or str(uuid4())
    run_token = _active_run_id.set(resolved_run_id)
    trace_token = _active_trace_id.set(None)
    try:
        yield resolved_run_id
    finally:
        _active_trace_id.reset(trace_token)
        _active_run_id.reset(run_token)


def get_logs(db_path: str | Path | None = None) -> List[Dict[str, Any]]:
    resolved = _ensure_db(db_path)
    with closing(sqlite3.connect(resolved)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                id,
                timestamp,
                version_tag,
                model,
                prompt,
                response,
                latency_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                run_id,
                trace_id,
                parent_trace_id,
                agent_name,
                step_name
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
    prompt_arg_name: str = "prompt",
    prompt_extractor: Callable[[tuple[Any, ...], Dict[str, Any], Dict[str, Any]], Any] | None = None,
    usage_extractor: Callable[[Any], Dict[str, int] | None] | None = None,
    agent_name: str | None = None,
    step_name: str | None = None,
) -> Callable[..., Any]:
    """Decorator that traces prompt-like function calls into SQLite.

    Supports both sync and async functions.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_signature = inspect.signature(func)
        resolved_step_name = step_name or func.__name__

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            current_run_id = _active_run_id.get()
            run_token = None
            if current_run_id is None:
                current_run_id = str(uuid4())
                run_token = _active_run_id.set(current_run_id)

            parent_trace_id = _active_trace_id.get()
            current_trace_id = str(uuid4())
            trace_token = _active_trace_id.set(current_trace_id)

            started = time.perf_counter()
            prompt_value = _extract_prompt_value(
                signature=func_signature,
                args=args,
                kwargs=kwargs,
                prompt_arg_name=prompt_arg_name,
                prompt_extractor=prompt_extractor,
            )
            call_prompt = _serialize(prompt_value)
            selected_model = _extract_model(model, kwargs)
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            response_payload = ""
            try:
                result = func(*args, **kwargs)
                response_payload = _serialize(result)
                token_usage = _resolve_token_usage(
                    result=result,
                    prompt_text=str(prompt_value),
                    response_text=str(result),
                    model_name=selected_model,
                    usage_extractor=usage_extractor,
                )
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
                    prompt_tokens=token_usage["prompt_tokens"],
                    completion_tokens=token_usage["completion_tokens"],
                    total_tokens=token_usage["total_tokens"],
                    run_id=current_run_id,
                    trace_id=current_trace_id,
                    parent_trace_id=parent_trace_id,
                    agent_name=agent_name,
                    step_name=resolved_step_name,
                )
                _active_trace_id.reset(trace_token)
                if run_token is not None:
                    _active_run_id.reset(run_token)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            current_run_id = _active_run_id.get()
            run_token = None
            if current_run_id is None:
                current_run_id = str(uuid4())
                run_token = _active_run_id.set(current_run_id)

            parent_trace_id = _active_trace_id.get()
            current_trace_id = str(uuid4())
            trace_token = _active_trace_id.set(current_trace_id)

            started = time.perf_counter()
            prompt_value = _extract_prompt_value(
                signature=func_signature,
                args=args,
                kwargs=kwargs,
                prompt_arg_name=prompt_arg_name,
                prompt_extractor=prompt_extractor,
            )
            call_prompt = _serialize(prompt_value)
            selected_model = _extract_model(model, kwargs)
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            response_payload = ""
            try:
                result = await func(*args, **kwargs)
                response_payload = _serialize(result)
                token_usage = _resolve_token_usage(
                    result=result,
                    prompt_text=str(prompt_value),
                    response_text=str(result),
                    model_name=selected_model,
                    usage_extractor=usage_extractor,
                )
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
                    prompt_tokens=token_usage["prompt_tokens"],
                    completion_tokens=token_usage["completion_tokens"],
                    total_tokens=token_usage["total_tokens"],
                    run_id=current_run_id,
                    trace_id=current_trace_id,
                    parent_trace_id=parent_trace_id,
                    agent_name=agent_name,
                    step_name=resolved_step_name,
                )
                _active_trace_id.reset(trace_token)
                if run_token is not None:
                    _active_run_id.reset(run_token)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    if _func is not None and callable(_func):
        return decorator(_func)
    return decorator
