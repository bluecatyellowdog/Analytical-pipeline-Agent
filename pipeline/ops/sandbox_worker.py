"""Runs one piece of custom cleaning code in its own process. Not imported by the pipeline.

Started by `sandbox.run_transform` as `python -I sandbox_worker.py <workdir>`, with an
environment that holds no secrets. It reads `input.parquet` and `code.py` from the
work directory, calls `transform(df)` and writes `output.parquet`. Anything that
goes wrong is printed as one line on stderr, prefixed `SANDBOX_ERROR:`.

Deliberately self-contained: it must not import the pipeline, because the code it
runs must not be able to reach the pipeline (or its database) either.
"""

from __future__ import annotations

import builtins
import math
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROW_ID = "_row_id"
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
REMOVED_BUILTINS = (
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "memoryview",
)
BLOCKED_EVENTS = (
    "socket.",
    "subprocess.",
    "os.system",
    "os.exec",
    "os.spawn",
    "os.posix_spawn",
    "os.startfile",
    "os.fork",
    "os.kill",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.rename",
    "os.mkdir",
    "os.chmod",
    "shutil.",
    "ctypes.",
    "winreg.",
    "urllib.",
    "http.",
    "webbrowser.",
)


def fail(message: str) -> None:
    print(f"SANDBOX_ERROR: {message}", file=sys.stderr)
    sys.exit(2)


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed in custom code")
    return builtins.__import__(name, globals, locals, fromlist, level)


def audit(event: str, args: tuple) -> None:
    """Defence in depth: refuse writes, network and process creation from user code."""
    if event == "open":
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
        if any(flag in mode for flag in "wax+"):
            raise PermissionError("custom code may not write files")
    elif event.startswith(BLOCKED_EVENTS):
        raise PermissionError(f"custom code may not do {event!r}")


def main() -> None:
    workdir = Path(sys.argv[1])
    code = (workdir / "code.py").read_text(encoding="utf-8")
    con = duckdb.connect(":memory:")
    df = con.execute(f"SELECT * FROM read_parquet('{(workdir / 'input.parquet').as_posix()}')").df()
    df = df.set_index(ROW_ID)
    input_ids = df.index

    safe_builtins = {k: v for k, v in vars(builtins).items() if k not in REMOVED_BUILTINS}
    safe_builtins["__import__"] = guarded_import
    namespace = {"__builtins__": safe_builtins, "pd": pd, "np": np, "re": re, "math": math}

    sys.addaudithook(audit)
    try:
        exec(compile(code, "<custom_code>", "exec"), namespace)  # noqa: S102 - this is the sandbox
        result = namespace["transform"](df.copy())
    except Exception as error:  # noqa: BLE001 - report any failure in the user's code
        fail(f"{type(error).__name__}: {error}")

    if not isinstance(result, pd.DataFrame):
        fail(f"transform must return a pandas DataFrame, got {type(result).__name__}")
    if not result.index.is_unique:
        fail("transform returned duplicate rows; custom code may not create rows")
    if not result.index.isin(input_ids).all():
        fail(
            "transform returned rows that were not in the input; keep the row index "
            "(do not reset it) and do not create rows"
        )

    out = result.reset_index(names=ROW_ID)
    con.register("result_df", out)
    con.execute(f"COPY (SELECT * FROM result_df) TO '{(workdir / 'output.parquet').as_posix()}'")


if __name__ == "__main__":
    main()
