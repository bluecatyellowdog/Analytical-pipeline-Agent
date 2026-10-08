"""Stage 1 end to end: claim verification, Gate 1 and the compact LLM profile."""

from __future__ import annotations

import json

import pytest

from pipeline.checks.gates import gate_1_ingestion, outcome_for
from pipeline.contracts.audit import CheckResult
from pipeline.contracts.context_brief import Claim, ColumnNote, ContextBrief
from pipeline.ingest import ingest, main
from pipeline.ingestion.claims import all_claims, derive_claims
from pipeline.ingestion.compact import compact_profile, render_profile_lines
from pipeline.ingestion.raw import RawStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    with RawStore.open() as store:
        yield store


@pytest.fixture
def brief() -> ContextBrief:
    return ContextBrief(
        description="Danish loan applications, one row per application.",
        goal="Understand which applicants default.",
        id_column="kunde_id",
        unit_of_observation="one application",
        column_notes=[ColumnNote(name="beloeb", description="Applied amount", unit="DKK")],
        excluded_variables=["email"],
        claims=[
            Claim(
                kind="value_range",
                column="alder",
                params={"min": 18, "max": 120},
                text="applicants are adults",
            ),
            Claim(kind="column_unique", column="dato", text="one application per day"),
        ],
    )


@pytest.fixture
def result(fixtures_dir, store, brief):
    return ingest(fixtures_dir / "danish_semicolon.csv", brief, store=store)


class TestStageOne:
    def test_gate_1_passes_on_a_clean_load(self, result) -> None:
        assert not result.blocked
        assert {c.name for c in result.gate.checks if not c.passed} == set()

    def test_the_raw_layer_is_untouched_by_profiling(self, result, store) -> None:
        store.verify(result.table)

    def test_the_manifest_records_the_code_that_ran(self, result) -> None:
        assert result.manifest.pipeline_version
        assert result.manifest.python_version
        assert result.manifest.seeds == {"profiling_sample": 42}
        assert result.manifest.inputs[0].file_hash == result.load_result.file.file_hash

    def test_gate_carries_every_flag_raised_upstream(self, result) -> None:
        codes = {f.code for f in result.gate.flags}
        assert "sentinel_candidate" in codes
        assert "pii_detected" in codes

    def test_profiling_is_repeatable(self, fixtures_dir, brief) -> None:
        """Same file, same numbers: the audit trail depends on it."""
        with RawStore.open() as first, RawStore.open() as second:
            a = ingest(fixtures_dir / "danish_semicolon.csv", brief, store=first)
            b = ingest(fixtures_dir / "danish_semicolon.csv", brief, store=second)
        assert a.profile.model_dump(exclude={"profiled_at"}) == b.profile.model_dump(
            exclude={"profiled_at"}
        )


class TestClaimVerification:
    def test_a_true_claim_passes(self, result) -> None:
        alder = next(r for r in result.claim_results if r.claim.column == "alder")
        assert alder.status == "pass"
        assert alder.observed["min"] == 19

    def test_a_false_claim_fails_with_what_was_observed(self, fixtures_dir, store) -> None:
        """A wrong brief is a finding: the point of verifying claims at all."""
        brief = ContextBrief(
            description="Loans",
            goal="Find defaults",
            claims=[
                Claim(
                    kind="value_range",
                    column="alder",
                    params={"min": 30, "max": 60},
                    text="applicants are 30-60",
                )
            ],
        )
        result = ingest(fixtures_dir / "danish_semicolon.csv", brief, store=store)
        alder = next(r for r in result.claim_results if r.claim.column == "alder")
        assert alder.status == "fail"
        assert alder.observed == {"min": 19.0, "max": 71.0}
        assert "19" in alder.message

    def test_a_uniqueness_claim_that_holds(self, result) -> None:
        kunde_id = next(
            r
            for r in result.claim_results
            if r.claim.column == "kunde_id" and r.claim.kind == "column_unique"
        )
        assert kunde_id.status == "pass"

    def test_a_missing_column_is_reported_with_the_real_columns(self, fixtures_dir, store) -> None:
        brief = ContextBrief(description="Loans", goal="Find defaults", target="default_12m")
        result = ingest(fixtures_dir / "danish_semicolon.csv", brief, store=store)
        target = next(r for r in result.claim_results if r.claim.column == "default_12m")
        assert target.status == "fail"
        assert "kunde_id" in target.message

    def test_an_unverifiable_claim_is_not_a_pass(self, fixtures_dir, store) -> None:
        """A range on a text column is neither true nor false; say so."""
        brief = ContextBrief(
            description="Loans",
            goal="Find defaults",
            claims=[Claim(kind="value_range", column="email", params={"min": 0, "max": 1})],
        )
        result = ingest(fixtures_dir / "danish_semicolon.csv", brief, store=store)
        email = next(r for r in result.claim_results if r.claim.column == "email")
        assert email.status == "not_checkable"

    def test_failed_claims_flag_the_gate_without_blocking(self, fixtures_dir, store) -> None:
        brief = ContextBrief(
            description="Loans",
            goal="Find defaults",
            claims=[Claim(kind="column_unique", column="alder", text="age is unique")],
        )
        result = ingest(fixtures_dir / "danish_semicolon.csv", brief, store=store)
        assert result.gate.outcome == "flag"
        assert not result.blocked, "a wrong brief is corrected by the human, not fatal"

    def test_structured_fields_become_claims(self) -> None:
        brief = ContextBrief(
            description="x",
            goal="y",
            target="t",
            id_column="k",
            excluded_variables=["koen"],
            column_notes=[ColumnNote(name="c", allowed_values=["0", "1"])],
        )
        derived = derive_claims(brief)
        assert ("column_unique", "k") in {(c.kind, c.column) for c in derived}
        assert ("allowed_values", "c") in {(c.kind, c.column) for c in derived}
        assert all(c.source == "derived" for c in derived)

    def test_a_stated_claim_is_not_duplicated_by_a_derived_one(self) -> None:
        brief = ContextBrief(
            description="x",
            goal="y",
            id_column="k",
            claims=[Claim(kind="column_unique", column="k", text="k is unique")],
        )
        unique_claims = [c for c in all_claims(brief) if c.kind == "column_unique"]
        assert len(unique_claims) == 1
        assert unique_claims[0].source == "brief"


class TestGateOne:
    def test_a_high_reject_rate_blocks(self, tmp_path, store) -> None:
        """Above 1% of lines unparseable, the file is not what it was taken to be."""
        rows = "\n".join(f"{i},name{i},{i * 10}" for i in range(50))
        path = tmp_path / "broken.csv"
        path.write_text(f"id,navn,beloeb\n{rows}\n1,too,many,fields\n", encoding="utf-8")

        result = ingest(path, store=store)
        assert result.blocked
        failed = next(c for c in result.gate.checks if not c.passed)
        assert failed.name == "reject_rate_within_threshold"

    def test_a_single_reject_in_a_large_file_only_flags(self, tmp_path, store) -> None:
        rows = "\n".join(f"{i},name{i},{i * 10}" for i in range(500))
        path = tmp_path / "mostly_fine.csv"
        path.write_text(f"id,navn,beloeb\n{rows}\n1,too,many,fields\n", encoding="utf-8")

        result = ingest(path, store=store)
        assert not result.blocked
        assert result.gate.outcome == "flag"

    def test_outcome_policy_is_worst_failure_wins(self) -> None:
        warn = CheckResult(name="w", layer="invariant", passed=False, severity="warn")
        block = CheckResult(name="b", layer="invariant", passed=False, severity="block")
        ok = CheckResult(name="o", layer="contract", passed=True)
        assert outcome_for([ok]) == "pass"
        assert outcome_for([ok, warn]) == "flag"
        assert outcome_for([ok, warn, block]) == "block"

    def test_checks_are_ordered_contract_then_invariant_then_judgment(self, result) -> None:
        """Spec 5.1: deterministic checks first, judgment second."""
        layers = [c.layer for c in result.gate.checks]
        assert layers == sorted(layers, key=["contract", "invariant", "judgment"].index)

    def test_a_profile_row_count_mismatch_blocks(self, result) -> None:
        tampered = result.profile.model_copy(update={"n_rows": 99})
        gate = gate_1_ingestion(result.load_result, tampered)
        assert gate.blocked
        assert not next(c for c in gate.checks if c.name == "profile_matches_loaded_rows").passed


class TestCompactProfile:
    def test_no_personal_value_reaches_the_llm_view(self, result) -> None:
        """The principle this whole module exists to enforce."""
        payload = json.dumps(compact_profile(result.profile, result.claim_results))
        for forbidden in ("anna.hansen", "@example.dk", "00417"):
            assert forbidden not in payload, f"{forbidden!r} leaked into the compact profile"

    def test_masked_examples_keep_their_shape(self, result) -> None:
        compact = compact_profile(result.profile)
        email = next(c for c in compact["columns"] if c["name"] == "email")
        assert email["pii"]["masked"] is True
        assert all("@" in example for example in email["examples"])
        assert all(char not in example for example in email["examples"] for char in "hns")

    def test_every_column_is_described(self, result) -> None:
        compact = compact_profile(result.profile)
        assert [c["name"] for c in compact["columns"]] == result.profile.column_names
        assert all(
            {"type", "coverage", "missing", "distinct"} <= set(c) for c in compact["columns"]
        )

    def test_claims_are_included_for_the_planner(self, result) -> None:
        compact = compact_profile(result.profile, result.claim_results)
        assert compact["brief_claims"]
        assert {c["status"] for c in compact["brief_claims"]} <= {
            "pass",
            "fail",
            "not_checkable",
        }

    def test_the_text_rendering_stays_compact(self, result) -> None:
        """Roughly one or two lines per column, so wide tables still fit a prompt."""
        text = render_profile_lines(result.profile)
        body = [line for line in text.splitlines() if line.startswith(("- ", "    "))]
        assert len(body) <= 3 * result.profile.n_columns + 10
        assert "anna.hansen" not in text


class TestCli:
    def test_writes_every_artifact(self, fixtures_dir, tmp_path, capsys) -> None:
        out = tmp_path / "run"
        code = main(
            [
                str(fixtures_dir / "danish_semicolon.csv"),
                "--out",
                str(out),
                "--db",
                str(tmp_path / "raw.duckdb"),
            ]
        )
        assert code == 0

        written = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
        assert written == {
            "REPORT.pdf",
            "details/1_analysis.pdf",
            "details/2_data_quality.pdf",
            "details/2_data_columns.xlsx",
            "details/3_decisions.pdf",
            "details/4_audit.pdf",
            "data/rejected_lines.csv",
            "logs/manifest.json",
            "logs/load_result.json",
            "logs/profile.json",
            "logs/profile_llm.json",
            "logs/claims.json",
            "logs/gate_1.json",
            "logs/profile.txt",
            "logs/ingestion_report.md",
        }
        for path in (out / "logs").glob("*.json"):
            json.loads(path.read_text(encoding="utf-8"))

        assert "Gate 1: PASS" in capsys.readouterr().out

    def test_reads_the_example_brief(self, fixtures_dir, repo_root, tmp_path, capsys) -> None:
        code = main(
            [
                str(fixtures_dir / "danish_semicolon.csv"),
                "--brief",
                str(repo_root / "examples" / "context_brief.yaml"),
                "--out",
                str(tmp_path / "run"),
            ]
        )
        output = capsys.readouterr().out
        assert code == 0
        assert "Brief claims:" in output
        assert "cust_id is unique" in output

    def test_a_blocked_run_exits_non_zero(self, tmp_path, capsys, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "broken.csv"
        rows = "\n".join(f"{i},x" for i in range(50))
        path.write_text(f"id,navn\n{rows}\n1,too,many\n", encoding="utf-8")

        assert main([str(path)]) == 1
        assert "Gate 1: BLOCK" in capsys.readouterr().out

    def test_without_out_every_run_lands_in_runs_and_the_index(
        self, fixtures_dir, tmp_path, capsys, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        main([str(fixtures_dir / "danish_semicolon.csv")])
        main([str(fixtures_dir / "latin1_aeoeaa.csv")])

        folders = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
        assert len(folders) == 2
        assert all((p / "REPORT.pdf").exists() for p in folders)

        index = (tmp_path / "runs" / "index.md").read_text(encoding="utf-8")
        assert "danish_semicolon.csv" in index
        assert "latin1_aeoeaa.csv" in index
        assert index.count("REPORT.pdf") == 2
        assert index.index("latin1_aeoeaa.csv") < index.index("danish_semicolon.csv")
