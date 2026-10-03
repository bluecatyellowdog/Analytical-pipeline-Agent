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

        store.connection.execute(
            f"CREATE TABLE {output_table} AS "
            f"SELECT o.{ROW_ID}, s._source_line, o.* EXCLUDE ({ROW_ID}) "
            f"FROM read_parquet('{output_path.as_posix()}') o "
            f"JOIN {source_table} s USING ({ROW_ID}) ORDER BY o.{ROW_ID}"
        )


def _error_from(stderr: str) -> str:
    for line in reversed(stderr.splitlines()):
        if line.startswith("SANDBOX_ERROR: "):
            return line.removeprefix("SANDBOX_ERROR: ")
    tail = stderr.strip().splitlines()[-1:] or ["no output"]
    return f"custom code process failed: {tail[0]}"


__all__ = ["SandboxError", "check_code", "run_transform", "sandbox_env"]
