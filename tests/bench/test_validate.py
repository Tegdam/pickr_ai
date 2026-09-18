from bench.capture.tokens import QwenTokenizer
from bench.capture.validate import compare, to_markdown
from tests.bench.test_tokens import StubHF


def _rec(n_words, role="agent", agent="ProductRecommendationAgent", prov="generated"):
    return {"call_role": role, "routed_agent": agent, "provenance": prov, "response_text": "x",
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": " ".join(["w"] * n_words)}]}


def test_compare_passes_when_quantiles_agree():
    tok = QwenTokenizer(StubHF(), "stub", "r")
    gen = [_rec(n) for n in range(90, 111)] * 2
    real = [_rec(n, prov="real") for n in range(95, 106)]
    rep = compare(gen, real, tok, tolerance=0.25, min_n=10)
    cell = rep["cells"]["A/agent"]
    assert cell["n_generated"] == 42 and cell["n_real"] == 11
    assert cell["status"] == "pass" and rep["overall_status"] == "pass"
    assert abs(cell["rel_diff_p50"]) < 0.05


def test_compare_fails_when_generated_is_far_off():
    tok = QwenTokenizer(StubHF(), "stub", "r")
    gen = [_rec(300) for _ in range(20)]
    real = [_rec(100, prov="real") for _ in range(20)]
    rep = compare(gen, real, tok)
    assert rep["cells"]["A/agent"]["status"] == "fail" and rep["overall_status"] == "fail"


def test_compare_marks_thin_real_cells_and_does_not_fail_on_them():
    tok = QwenTokenizer(StubHF(), "stub", "r")
    gen = [_rec(100) for _ in range(20)] + [_rec(50, role="guardrail_input") for _ in range(20)]
    real = [_rec(100, prov="real") for _ in range(12)] + [_rec(500, role="guardrail_input", prov="real") for _ in range(3)]
    rep = compare(gen, real, tok, min_n=10)
    assert rep["cells"]["A/guardrail_input"]["status"] == "insufficient_real"
    assert rep["overall_status"] == "pass"


def test_real_records_without_routing_are_assigned_by_call_role_only():
    tok = QwenTokenizer(StubHF(), "stub", "r")
    gen = [_rec(100, agent="ReviewSummarizationAgent") for _ in range(20)]      # B in generated
    real = [_rec(100, agent=None, prov="real") for _ in range(20)]               # unknown agent -> A
    rep = compare(gen, real, tok)
    assert "B/agent" in rep["cells"] and rep["cells"]["B/agent"]["n_real"] == 0
    assert rep["cells"]["A/agent"]["n_real"] == 20
    assert "routed_agent is unknown for real traffic" in rep["notes"][0]


def test_overall_is_insufficient_when_no_cell_can_be_judged():
    tok = QwenTokenizer(StubHF(), "stub", "r")
    rep = compare([_rec(10)], [_rec(10, prov="real")], tok)      # 1 real record < min_n
    assert rep["overall_status"] == "insufficient"


def test_markdown_has_a_row_per_cell():
    tok = QwenTokenizer(StubHF(), "stub", "r")
    rep = compare([_rec(10)], [_rec(10, prov="real")], tok)
    md = to_markdown(rep)
    assert "| A/agent |" in md and "overall" in md.lower()
