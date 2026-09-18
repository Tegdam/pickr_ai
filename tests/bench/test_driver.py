import json

from app import agents
from app.conversation import HISTORY_WINDOW

from bench.capture.generator import Conversation, GeneratedQuery
from bench.capture.driver import capture_all, read_raw, run_conversation, run_single_turn
from bench.capture.recorder import Recorder, RouteCapture


def _q(qid, text, intent="review", phrasing="keyword"):
    return GeneratedQuery(qid, text, intent, phrasing)


def test_single_turn_records_guardrail_agent_guardrail(fake_chat, small_catalog):
    coordinator = agents.CoordinatorAgent()
    with Recorder() as rec, RouteCapture.installed() as routes:
        records = run_single_turn(coordinator, _q("q1", "What do the reviews say about Alpha Laptop?"), rec, routes)
    roles = [r.call_role for r in records]
    assert roles == ["guardrail_input", "agent", "guardrail_output"]
    assert all(r.query_id == "q1" and r.routed_agent == "ReviewSummarizationAgent" for r in records)
    assert records[0].routed_via == "keyword" and records[0].route_status == "ok"
    assert records[0].intent == "review" and records[0].phrasing == "keyword"
    assert records[0].query_text == "What do the reviews say about Alpha Laptop?"


def test_natural_phrasing_goes_through_classifier(fake_chat, small_catalog):
    fake_chat.classifier_category = "review"
    coordinator = agents.CoordinatorAgent()
    with Recorder() as rec, RouteCapture.installed() as routes:
        records = run_single_turn(coordinator, _q("q2", "What do customers think of Alpha Laptop?", phrasing="natural"), rec, routes)
    assert [r.call_role for r in records] == ["guardrail_input", "classifier", "agent", "guardrail_output"]
    assert records[0].routed_via == "llm_fallback"


def test_non_llm_agent_records_only_guardrail_input(fake_chat, small_catalog):
    coordinator = agents.CoordinatorAgent()
    with Recorder() as rec, RouteCapture.installed() as routes:
        records = run_single_turn(coordinator, _q("q3", "Is Alpha Laptop in stock?", intent="stock"), rec, routes)
    assert [r.call_role for r in records] == ["guardrail_input"]
    assert records[0].routed_agent == "StockAvailabilityAgent"


def test_conversation_condenses_from_turn_one_and_windows_history(fake_chat, small_catalog):
    turns = tuple(_q(f"c1-t{i}", t, intent="followup_review", phrasing="natural") for i, t in enumerate([
        "What do the reviews say about Alpha Laptop?",
        "Is it in stock?", "Can I return it?", "What about the warranty?", "Is Beta Laptop in stock?",
    ]))
    turns = (_q("c1-t0", turns[0].text),) + turns[1:]
    conv = Conversation("c1", "medium", turns)
    coordinator = agents.CoordinatorAgent()
    with Recorder() as rec, RouteCapture.installed() as routes:
        records = run_conversation(coordinator, conv, rec, routes)
    by_turn = {}
    for r in records:
        by_turn.setdefault(r.turn_index, []).append(r)
    assert set(by_turn) == {0, 1, 2, 3, 4}
    assert by_turn[0][0].call_role == "guardrail_input"          # no condense on turn 0
    assert all(by_turn[t][0].call_role == "condense" for t in (1, 2, 3, 4))
    assert all(r.conversation_id == "c1" for r in records)
    # the condense prompt at turn 4 carries at most HISTORY_WINDOW rows of transcript
    transcript = by_turn[4][0].messages[1]["content"].split("\n\nFollow-up message: ")[0]
    rows = [l for l in transcript.splitlines() if l.startswith(("Customer: ", "Assistant: "))]
    assert len(rows) == HISTORY_WINDOW          # 8 rows of history exist by turn 4; only the last 6 are sent
    # the resolved (condensed) query is what the coordinator saw
    assert by_turn[1][1].call_role == "guardrail_input"
    assert by_turn[1][1].messages[1]["content"] == "Is it in stock?"  # guardrail input sees the RAW query
    assert by_turn[1][0].query_text == "Is it in stock?"


def test_capture_all_writes_jsonl_and_resumes(fake_chat, small_catalog, tmp_path):
    out = tmp_path / "raw.jsonl"
    qs = [_q("q1", "What do the reviews say about Alpha Laptop?"), _q("q2", "Is Alpha Laptop in stock?", intent="stock")]
    convs = [Conversation("c1", "shallow", (_q("c1-t0", "What do the reviews say about Alpha Laptop?"),
                                              _q("c1-t1", "Is Alpha Laptop in stock?", "followup_review", "natural")))]
    n1, f1 = capture_all(qs, convs, out, workers=2, app_git_sha="abc")
    rows = read_raw(out)
    # q1: 3 calls, q2: 1, c1-t0: 3, c1-t1: condense + guardrail_input = 2
    assert n1 == len(rows) == 9 and f1 == 0
    assert all(r["app_git_sha"] == "abc" and r["provenance"] == "generated" for r in rows)
    n2, f2 = capture_all(qs, convs, out, workers=2, app_git_sha="abc")
    assert n2 == 0 and f2 == 0 and len(read_raw(out)) == len(rows)
    # every line is valid JSON with the contract's keys
    required = {"record_id", "query_id", "conversation_id", "turn_index", "call_index", "call_role", "intent",
                "phrasing", "query_text", "routed_agent", "routed_via", "route_status", "model", "messages",
                "response_format", "temperature", "max_tokens", "response_text", "finish_reason",
                "prompt_tokens_openai", "completion_tokens_openai", "latency_ms", "captured_at", "provenance", "app_git_sha"}
    for line in out.read_text().splitlines():
        assert required <= set(json.loads(line))


def test_capture_all_resumes_only_pending_units(fake_chat, small_catalog, tmp_path):
    out = tmp_path / "raw.jsonl"
    qs = [_q("q1", "What do the reviews say about Alpha Laptop?"), _q("q2", "Is Alpha Laptop in stock?", intent="stock")]
    convs = [Conversation("c1", "shallow", (_q("c1-t0", "What do the reviews say about Alpha Laptop?"),
                                              _q("c1-t1", "Is Alpha Laptop in stock?", "followup_review", "natural")))]
    capture_all(qs[:1], [], out, workers=1, app_git_sha="abc")          # q1 done in a previous run
    assert {r["query_id"] for r in read_raw(out)} == {"q1"}
    n, f = capture_all(qs, convs, out, workers=2, app_git_sha="abc")    # re-run the same command
    rows = read_raw(out)
    assert n == 6 and f == 0 and len(rows) == 9                          # q2 (1) + c1 (3 + 2) added; q1 not repeated
    assert sum(r["query_id"] == "q1" for r in rows) == 3


def test_capture_all_isolates_a_failing_unit_and_writes_nothing_for_it(fake_chat, small_catalog, tmp_path):
    # NOTE: deviates from the brief's Beta Laptop example. Beta Laptop (P2) has
    # no reviews in the fixture, so ReviewSummarizationAgent returns "No reviews
    # found" without ever calling the LLM -- the only call that mentions "Beta
    # Laptop" is the guardrail_input classify call, and check_input fails OPEN
    # on any classifier exception (see app/guardrails.py), so raising there
    # never propagates and the unit does not fail. Gamma Phone (P3) has one
    # review, so its query reaches ReviewSummarizationAgent's uncaught
    # client.chat.completions.create call -- that's the call whose exception
    # actually propagates out of CoordinatorAgent.handle_query (try/finally,
    # no except) and fails the unit. Intent (one unit raises, nothing is
    # written for it, resume re-runs it) is unchanged.
    original = fake_chat.create

    def failing_create(**kwargs):
        if "Gamma Phone" in kwargs["messages"][-1]["content"]:
            raise RuntimeError("simulated API failure")
        return original(**kwargs)

    fake_chat.create = failing_create
    out = tmp_path / "raw.jsonl"
    qs = [_q("q1", "What do the reviews say about Alpha Laptop?"),
          _q("q2", "What do the reviews say about Gamma Phone?")]      # fails inside the agent call
    written, failures = capture_all(qs, [], out, workers=2, app_git_sha="abc")
    rows = read_raw(out)
    assert failures == 1 and written == len(rows)
    assert {r["query_id"] for r in rows} == {"q1"}                       # nothing partial for q2
    fake_chat.create = original
    written2, failures2 = capture_all(qs, [], out, workers=1, app_git_sha="abc")
    assert failures2 == 0 and {r["query_id"] for r in read_raw(out)} == {"q1", "q2"}   # q2 re-runs on resume
