import json

from bench.capture import cli
from bench.capture.generator import from_json


def test_generate_writes_queries_json(small_catalog, tmp_path, monkeypatch):
    from bench.capture.generator import Catalog
    monkeypatch.setattr(cli.Catalog, "from_app", classmethod(lambda c: Catalog(small_catalog.products, small_catalog.reviews, small_catalog.policies)))
    out = tmp_path / "queries.json"
    rc = cli.main(["generate", "--seed", "3", "--n-single", "20", "--n-per-profile", "2", "--out", str(out)])
    assert rc == 0
    qs, convs = from_json(json.loads(out.read_text()))
    assert len(qs) == 20 and len(convs) == 6


def test_capture_then_export_end_to_end(fake_chat, small_catalog, tmp_path, monkeypatch):
    from bench.capture.generator import Catalog
    from bench.capture.tokens import QwenTokenizer
    from tests.bench.test_tokens import StubHF
    monkeypatch.setattr(cli.Catalog, "from_app", classmethod(lambda c: Catalog(small_catalog.products, small_catalog.reviews, small_catalog.policies)))
    monkeypatch.setattr(cli.QwenTokenizer, "load", classmethod(lambda c, **kw: QwenTokenizer(StubHF(), "stub", "r")))
    q = tmp_path / "queries.json"
    raw = tmp_path / "raw.jsonl"
    traces = tmp_path / "traces"
    (traces / "schemas").mkdir(parents=True)
    (traces / "schemas" / "product_card.schema.json").write_text("{}")
    assert cli.main(["generate", "--seed", "1", "--n-single", "10", "--n-per-profile", "1", "--out", str(q)]) == 0
    assert cli.main(["capture", "--queries", str(q), "--out", str(raw), "--workers", "2"]) == 0
    assert raw.read_text().count("\n") > 10
    assert cli.main(["export", "--raw", str(raw), "--out-dir", str(traces), "--version", "1", "--seed", "1"]) == 0
    assert (traces / "chat_v1.jsonl").exists() and (traces / "multiturn_deep_v1.meta.json").exists()
    # immutability: a second export at the same version fails loudly
    assert cli.main(["export", "--raw", str(raw), "--out-dir", str(traces), "--version", "1", "--seed", "1"]) == 1


def test_validate_writes_markdown(tmp_path, monkeypatch):
    from bench.capture.tokens import QwenTokenizer
    from tests.bench.test_tokens import StubHF
    monkeypatch.setattr(cli.QwenTokenizer, "load", classmethod(lambda c, **kw: QwenTokenizer(StubHF(), "stub", "r")))
    rec = {"call_role": "agent", "routed_agent": "X", "provenance": "generated", "response_text": "x",
           "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "a b c"}]}
    gen, real = tmp_path / "g.jsonl", tmp_path / "r.jsonl"
    gen.write_text(json.dumps(rec) + "\n")
    real.write_text(json.dumps({**rec, "provenance": "real", "routed_agent": None}) + "\n")
    out = tmp_path / "v.md"
    # its single real record is below min_n, so overall is "insufficient" -> exit 1
    assert cli.main(["validate", "--generated", str(gen), "--real", str(real), "--out", str(out)]) == 1
    assert "| A/agent |" in out.read_text()


def test_validate_refuses_missing_inputs(tmp_path):
    present = tmp_path / "g.jsonl"
    present.write_text("")
    assert cli.main(["validate", "--generated", str(present), "--real", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "v.md")]) == 1
    assert not (tmp_path / "v.md").exists()


def test_git_sha_appends_dirty_suffix_when_tree_is_dirty(monkeypatch):
    def fake_check_output(args, text=True):
        if args[:2] == ["git", "rev-parse"]:
            return "deadbeef\n"
        if args[:2] == ["git", "status"]:
            return " M bench/capture/cli.py\n"
        raise AssertionError(args)

    monkeypatch.setattr(cli.subprocess, "check_output", fake_check_output)
    assert cli.git_sha() == "deadbeef-dirty"


def test_git_sha_has_no_suffix_when_tree_is_clean(monkeypatch):
    def fake_check_output(args, text=True):
        if args[:2] == ["git", "rev-parse"]:
            return "deadbeef\n"
        if args[:2] == ["git", "status"]:
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(cli.subprocess, "check_output", fake_check_output)
    assert cli.git_sha() == "deadbeef"


def test_git_sha_returns_none_on_subprocess_failure(monkeypatch):
    def raising(*a, **kw):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(cli.subprocess, "check_output", raising)
    assert cli.git_sha() is None


def test_langsmith_returns_1_when_nothing_exported(tmp_path, monkeypatch):
    import bench.capture.langsmith_export as ls
    monkeypatch.setattr(ls, "fetch_llm_runs", lambda project_name, since=None, client=None: [])
    assert cli.main(["langsmith", "--project", "p", "--out", str(tmp_path / "ls.jsonl")]) == 1


def test_export_reports_empty_raw_as_refusal(tmp_path, monkeypatch):
    from bench.capture.tokens import QwenTokenizer
    from tests.bench.test_tokens import StubHF
    monkeypatch.setattr(cli.QwenTokenizer, "load", classmethod(lambda c, **kw: QwenTokenizer(StubHF(), "stub", "r")))
    traces = tmp_path / "traces"
    (traces / "schemas").mkdir(parents=True)
    (traces / "schemas" / "product_card.schema.json").write_text("{}")
    assert cli.main(["export", "--raw", str(tmp_path / "missing.jsonl"), "--out-dir", str(traces), "--version", "2", "--seed", "1"]) == 1
    assert not any(traces.glob("*_v2.*"))
