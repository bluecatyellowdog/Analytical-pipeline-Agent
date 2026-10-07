"""Where custom cleaning code runs: checked first, then executed in its own process.

Two layers, because neither is enough alone:

1. A static check refuses code before it runs: imports outside a short allowlist,
   file and network access, `eval`/`exec`, dunder attributes, pandas readers and
   writers.
2. The code runs in a separate Python process (`sandbox_worker.py`) with an
   environment that holds no secrets, restricted builtins, an audit hook that
   blocks file writes, network and process creation, and a timeout.

This is a guard rail, not a security boundary: a determined attacker can escape a
Python-level sandbox. The spec's container sandbox (no network, resource limits)
is the hard boundary and is recorded as a follow-up in DECISIONS.md. The code the
agent writes only ever sees a masked profile, never data, so the realistic risk is
a mistaken transform, which the accounting in `apply_op` catches regardless.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pipeline.ingestion.raw import ROW_ID, RawStore, quote_ident

TIMEOUT_SECONDS = 120
WORKER = Path(__file__).with_name("sandbox_worker.py")

ALLOWED_IMPORTS = frozenset(
    {
        "pandas",
        "numpy",
        "re",
        "math",
        "datetime",
        "decimal",
        "string",
        "unicodedata",
        "collections",
        "itertools",
        "functools",
    }
)
FORBIDDEN_NAMES = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
    }
)
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        # pandas writers
        "to_csv",
        "to_parquet",
        "to_pickle",
        "to_sql",
        "to_excel",
        "to_json",
        "to_feather",
        "to_hdf",
        "to_stata",
        "to_html",
        "to_latex",
        "to_xml",
        "to_clipboard",
        "to_orc",
        # numpy file access
        "load",
        "save",
        "savez",
        "savez_compressed",
        "fromfile",
        "tofile",
        "loadtxt",
        "savetxt",
        "genfromtxt",
        "memmap",
        "ctypeslib",
    }
)
SAFE_ENV_VARIABLES = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "LANG", "LC_ALL")


class SandboxError(Exception):
    """The code was refused, crashed, timed out or broke the row contract."""


def check_code(code: str) -> list[str]:
    """Every reason this code may not run. Empty means it may."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return [f"syntax error on line {error.lineno}: {error.msg}"]

    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if getattr(node, "level", 0) or name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(
                        f"import of {name!r} is not allowed (allowed: {sorted(ALLOWED_IMPORTS)})"
                    )
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"{node.id!r} is not allowed in custom code")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                problems.append(f"dunder attribute {node.attr!r} is not allowed")
            elif node.attr in FORBIDDEN_ATTRIBUTES or node.attr.startswith("read_"):
                problems.append(
                    f"{node.attr!r} is not allowed: custom code may not read or write files"
                )
        elif isinstance(node, ast.Global | ast.Nonlocal):
            problems.append("global/nonlocal statements are not allowed")

    has_transform = any(
        isinstance(node, ast.FunctionDef) and node.name == "transform" for node in tree.body
    )
    if not has_transform:
        problems.append("custom code must define a top-level function transform(df) -> DataFrame")
    return problems


def sandbox_env() -> dict[str, str]:
    """Only what Python needs to start. No API keys, no credentials."""
    return {k: os.environ[k] for k in SAFE_ENV_VARIABLES if k in os.environ}


def run_transform(
    store: RawStore, source_table: str, columns: list[str], code: str, output_table: str
) -> None:
    """Run `transform(df)` on `source_table` and store the result as `output_table`.

    The output keeps `_row_id` from the code's result and takes `_source_line` from
    the source, so the code can neither invent rows nor rewrite provenance. Column
    checks (declared vs actual) are the caller's job.
    """
    problems = check_code(code)
    if problems:
        raise SandboxError("; ".join(problems))

    selection = ", ".join([ROW_ID, *(quote_ident(c) for c in columns)])
    with tempfile.TemporaryDirectory(prefix="pipeline_sandbox_") as tmp:
        workdir = Path(tmp)
        input_path, output_path = workdir / "input.parquet", workdir / "output.parquet"
        (workdir / "code.py").write_text(code, encoding="utf-8")
        store.connection.execute(
            f"COPY (SELECT {selection} FROM {source_table} ORDER BY {ROW_ID}) "
            f"TO '{input_path.as_posix()}' (FORMAT parquet)"
        )

        try:
            completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
                [sys.executable, "-I", str(WORKER), str(workdir)],
                cwd=workdir,
                env=sandbox_env(),
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise SandboxError(
                f"custom code ran out of time ({TIMEOUT_SECONDS} s) and was stopped"
            ) from error

        if completed.returncode != 0 or not output_path.exists():
            raise SandboxError(_error_from(completed.stderr))

        output = f"read_parquet('{output_path.as_posix()}')"
        columns_sql = ", ".join(
            _restored_type(store, output, source_table, name, out_type)
            for name, out_type in _types(store, f"SELECT * FROM {output}").items()
            if name != ROW_ID
        )
        store.connection.execute(
            f"CREATE TABLE {output_table} AS "
            f"SELECT o.{ROW_ID}, s._source_line, {columns_sql} "
            f"FROM {output} o "
            f"JOIN {source_table} s USING ({ROW_ID}) ORDER BY o.{ROW_ID}"
        )


_INTEGER_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT")
_INTEGER_TYPES += ("UINTEGER", "UBIGINT")


def _round_trip_shift(source_type: str, output_type: str) -> bool:
    """Type pairs pandas produces on its own: DATE comes back as a timestamp, and so on."""
    source, output = source_type.upper(), output_type.upper()
    if source == output:
        return False
    if source in ("DATE",) or source.startswith("TIMESTAMP"):
        return output.startswith("TIMESTAMP")
    if source.startswith(_INTEGER_TYPES):
        return output.startswith(_INTEGER_TYPES) or output in ("DOUBLE", "FLOAT")
    if source.startswith("DECIMAL") or source in ("FLOAT", "REAL"):
        return output in ("DOUBLE", "FLOAT")
    return False


def _restored_type(
    store: RawStore, output: str, source_table: str, name: str, output_type: str
) -> str:
    """A column's own type back, when only the trip through pandas changed it.

    The code ran on a pandas DataFrame, which has no DATE type and turns integer
    columns with gaps into floats. Without this, every column would come back
    subtly retyped and be refused as an undeclared change. The original type is
    restored only when no value changes by it; a real type change stays, and must
    be declared like any other.
    """
    column = quote_ident(name)
    source_type = _types(store, f"SELECT * FROM {source_table}").get(name)
    if source_type is None or not _round_trip_shift(source_type, output_type):
        return f"o.{column}"
    lossy = store.connection.execute(
        f"SELECT count(*) FROM {output} WHERE {column} IS NOT NULL AND "
        f"(TRY_CAST({column} AS {source_type}) IS NULL OR "
        f"CAST(TRY_CAST({column} AS {source_type}) AS {output_type}) IS DISTINCT FROM {column})"
    ).fetchone()[0]
    return f"o.{column}" if lossy else f"CAST(o.{column} AS {source_type}) AS {column}"


def _types(store: RawStore, query: str) -> dict[str, str]:
    return {row[0]: row[1] for row in store.connection.execute(f"DESCRIBE {query}").fetchall()}


def _error_from(stderr: str) -> str:
    for line in reversed(stderr.splitlines()):
        if line.startswith("SANDBOX_ERROR: "):
            return line.removeprefix("SANDBOX_ERROR: ")
    tail = stderr.strip().splitlines()[-1:] or ["no output"]
    return f"custom code process failed: {tail[0]}"


__all__ = ["SandboxError", "check_code", "run_transform", "sandbox_env"]
