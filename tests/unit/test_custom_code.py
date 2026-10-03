"""Custom code: free to transform, never free to hide what it did.

An agent may write its own pandas code when no library op fits. These tests pin
down what makes that safe: the code runs in a separate process, must declare
every column it changes, adds or drops, may not create rows, may not drop rows
or create nulls without saying why, and is replayed exactly like any other op.
"""

from __future__ import annotations

import pytest

from pipeline.checks.reconciliation import reconcile
from pipeline.ingestion.raw import RawStore
from pipeline.ops import OpContext, OpError, apply_op, finalize, sandbox
from pipeline.ops import custom as custom_module

RUN_ID = "run_custom"

NORMALIZE_BELOEB = """
def transform(df):
    df["beloeb"] = (
        df["beloeb"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    ).astype(float)
    return df
"""


@pytest.fixture
def store():
    with RawStore.open() as store:
        yield store


@pytest.fixture
def raw(store):
    return store.create_text_table(
        "t",
        ["id", "alder", "beloeb", "by"],
        [
            ["001", "31", "12.345,50", "Nord"],
            ["002", "NA", "999", "nord"],
            ["003", "54", "7.800,00", "SYD"],
            ["004", "23", "250.000,00", "syd"],
        ],
        [2, 3, 4, 5],
    )


@pytest.fixture
def ctx(store, raw):
    return OpContext.from_raw(store, raw, RUN_ID)


def custom(ctx, code, rationale="test", **declared):
    return apply_op(ctx, "custom_code", {"code": code, **declared}, rationale=rationale)


class TestDeclaredChanges:
    def test_a_declared_change_runs_and_is_recorded(self, store, ctx) -> None:
        new, record = custom(ctx, NORMALIZE_BELOEB, changes=["beloeb"])

        assert store.column_values(new.table, "beloeb") == [12345.5, 999.0, 7800.0, 250000.0]
        assert record.op_name == "custom_code"
        assert record.code.strip() == NORMALIZE_BELOEB.strip()
        assert record.cells_changed == {"beloeb": 4}
        assert record.target_columns == ["beloeb"]

    def test_an_undeclared_change_is_refused(self, ctx) -> None:
        code = NORMALIZE_BELOEB.replace(
            "return df", 'df["by"] = df["by"].str.upper()\n    return df'
        )
        with pytest.raises(OpError, match="undeclared.*by"):
            custom(ctx, code, changes=["beloeb"])

    def test_new_nulls_must_be_declared(self, ctx) -> None:
        code = """
def transform(df):
    df["alder"] = pd.to_numeric(df["alder"].where(df["alder"] != "NA"))
    return df
"""
        with pytest.raises(OpError, match="nulls"):
            custom(ctx, code, changes=["alder"])

    def test_declared_nulls_pass_and_are_flagged(self, ctx) -> None:
        code = """
def transform(df):
    df["alder"] = pd.to_numeric(df["alder"].where(df["alder"] != "NA"))
    return df
"""
        _, record = custom(
            ctx,
            code,
            changes=["alder"],
            may_create_nulls=["alder"],
            null_reason="'NA' means the age was not recorded.",
        )
        assert record.new_nulls == {"alder": 1}
        assert any(f.code == "custom_code_new_nulls" for f in record.flags)

    def test_a_null_declaration_needs_a_reason(self, ctx) -> None:
        with pytest.raises(OpError, match="null_reason"):
            custom(ctx, NORMALIZE_BELOEB, changes=["beloeb"], may_create_nulls=["beloeb"])


class TestRowsAndColumns:
    def test_dropping_rows_without_declaring_it_is_refused(self, ctx) -> None:
        code = "def transform(df):\n    return df[df['by'] != 'nord']\n"
        with pytest.raises(OpError, match="drop"):
            custom(ctx, code)

    def test_declared_row_drops_are_captured_with_their_reason(self, store, ctx) -> None:
        code = "def transform(df):\n    return df[df['by'] != 'nord']\n"
        new, record = custom(ctx, code, drop_reason="test rows from the old system")
        assert record.rows_dropped == 1
        reason = store.connection.execute(f"SELECT reason FROM {new.dropped_table}").fetchone()
        assert reason == ("test rows from the old system",)

    def test_creating_rows_is_refused(self, ctx) -> None:
        code = "def transform(df):\n    return pd.concat([df, df.iloc[:1]])\n"
        with pytest.raises(OpError, match="row"):
            custom(ctx, code)

    def test_resetting_the_row_index_is_refused(self, ctx) -> None:
        code = "def transform(df):\n    return df.reset_index(drop=True)\n"
        with pytest.raises(OpError, match="row"):
            custom(ctx, code)

    def test_a_declared_new_column_is_added(self, store, ctx) -> None:
        code = "def transform(df):\n    df['by_norm'] = df['by'].str.title()\n    return df\n"
        new, _ = custom(ctx, code, adds=["by_norm"])
        assert new.columns == ["id", "alder", "beloeb", "by", "by_norm"]
        assert store.column_values(new.table, "by_norm") == ["Nord", "Nord", "Syd", "Syd"]

    def test_an_undeclared_new_column_is_refused(self, ctx) -> None:
        code = "def transform(df):\n    df['extra'] = 1\n    return df\n"
        with pytest.raises(OpError, match="extra"):
            custom(ctx, code)

    def test_a_declared_dropped_column_is_removed(self, ctx) -> None:
        code = "def transform(df):\n    return df.drop(columns=['by'])\n"
        new, _ = custom(ctx, code, drops=["by"])
        assert "by" not in new.columns

    def test_declaring_a_column_that_does_not_exist_is_refused(self, ctx) -> None:
        with pytest.raises(OpError, match="nope"):
            custom(ctx, NORMALIZE_BELOEB, changes=["beloeb", "nope"])


class TestSandbox:
    @pytest.mark.parametrize(
        ("code", "reason"),
        [
            ("import os\ndef transform(df):\n    return df\n", "import"),
            ("import subprocess\ndef transform(df):\n    return df\n", "import"),
            ("def transform(df):\n    open('x.txt', 'w')\n    return df\n", "open"),
            ("def transform(df):\n    return eval('df')\n", "eval"),
            ("def transform(df):\n    df.to_csv('leak.csv')\n    return df\n", "to_csv"),
            ("def transform(df):\n    return pd.read_csv('C:/secret.csv')\n", "read_csv"),
            ("def transform(df):\n    return df.__class__\n", "__class__"),
            ("x = 1\n", "transform"),
        ],
    )
    def test_forbidden_code_is_refused_before_it_runs(self, ctx, code, reason) -> None:
        with pytest.raises(OpError, match=reason):
            custom(ctx, code)

    def test_a_crash_in_the_code_is_reported(self, ctx) -> None:
        code = "def transform(df):\n    raise ValueError('boom')\n"
        with pytest.raises(OpError, match="boom"):
            custom(ctx, code)

    def test_returning_something_else_than_a_dataframe_is_refused(self, ctx) -> None:
        with pytest.raises(OpError, match="DataFrame"):
            custom(ctx, "def transform(df):\n    return 1\n")

    def test_a_runaway_loop_is_stopped(self, ctx, monkeypatch) -> None:
        monkeypatch.setattr(sandbox, "TIMEOUT_SECONDS", 8)
        code = "def transform(df):\n    while True:\n        pass\n"
        with pytest.raises(OpError, match="time"):
            custom(ctx, code)

    def test_the_sandbox_gets_no_secrets_from_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-secret")
        assert "LLM_API_KEY" not in sandbox.sandbox_env()


class TestAuditTrail:
    def test_custom_code_mixed_with_library_ops_reconciles_and_replays(
        self, store, raw, ctx
    ) -> None:
        records = []
        ctx, record = apply_op(ctx, "tokens_to_null", {"columns": ["alder"], "tokens": ["NA"]})
        records.append(record)
        ctx, record = custom(ctx, NORMALIZE_BELOEB, changes=["beloeb"])
        records.append(record)
        code = "def transform(df):\n    df['by'] = df['by'].str.title()\n    return df\n"
        ctx, record = custom(ctx, code, changes=["by"])
        records.append(record)

        final = finalize(ctx)
        checks = reconcile(
            store, raw, final, records, run_id=RUN_ID, dropped_table=ctx.dropped_table
        )
        failed = [c for c in checks if not c.passed and c.severity == "block"]
        assert not failed, failed
        assert any(c.name == "replay_reproduces_output" and c.passed for c in checks)

    def test_raw_is_untouched(self, store, raw, ctx) -> None:
        custom(ctx, NORMALIZE_BELOEB, changes=["beloeb"])
        store.verify(raw)


class TestSecondLayer:
    """The worker's own guards, with the static check switched off."""

    @pytest.fixture(autouse=True)
    def no_static_check(self, monkeypatch):
        monkeypatch.setattr(sandbox, "check_code", _transform_only)
        monkeypatch.setattr(custom_module, "check_code", _transform_only)

    def test_imports_are_refused_at_runtime(self, ctx) -> None:
        code = "import os\ndef transform(df):\n    return df\n"
        with pytest.raises(OpError, match="not allowed"):
            custom(ctx, code)

    def test_file_writes_are_refused_at_runtime(self, ctx, tmp_path) -> None:
        target = (tmp_path / "leak.csv").as_posix()
        code = f"def transform(df):\n    df.to_csv('{target}')\n    return df\n"
        with pytest.raises(OpError, match="may not write"):
            custom(ctx, code)
        assert not (tmp_path / "leak.csv").exists()

    def test_open_is_not_available_at_runtime(self, ctx) -> None:
        code = "def transform(df):\n    open('x.txt', 'w')\n    return df\n"
        with pytest.raises(OpError, match="open"):
            custom(ctx, code)


def _transform_only(code: str) -> list[str]:
    return [] if "def transform" in code else ["no transform"]
