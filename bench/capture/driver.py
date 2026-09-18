"""Drives Pickr's real request path in-process and writes raw call records.

Single-turn queries go straight through CoordinatorAgent.handle_query.
Conversations replicate app.conversation.handle_conversational_query exactly
— condense against the last HISTORY_WINDOW rows, route the resolved query,
append the exchange — but with an in-memory history instead of RDS, so no
database is needed and nothing is persisted to the app's tables.
"""
from __future__ import annotations

import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.agents import CoordinatorAgent
from app.conversation import HISTORY_WINDOW, condense_query
from app.models import UserQuery

from .generator import Conversation, GeneratedQuery
from .recorder import QUERY_CONTEXT, CallRecord, QueryContext, Recorder, RouteCapture


def _stamp(records: list[CallRecord], q: GeneratedQuery, route: dict | None, app_git_sha: str | None) -> list[CallRecord]:
    for r in records:
        r.intent = q.intent
        r.phrasing = q.phrasing
        r.query_text = q.text
        r.routed_agent = route.get("agent") if route else None
        r.routed_via = route.get("via") if route else None
        r.route_status = route.get("status") if route else None
        r.app_git_sha = app_git_sha
    return records


def _records_for(recorder: Recorder, query_id: str) -> list[CallRecord]:
    with recorder._lock:
        return [r for r in recorder.records if r.query_id == query_id]


def run_single_turn(coordinator: CoordinatorAgent, query: GeneratedQuery, recorder: Recorder,
                    routes: RouteCapture, app_git_sha: str | None = None) -> list[CallRecord]:
    QUERY_CONTEXT.set(QueryContext(query.query_id, None, 0))
    coordinator.handle_query(UserQuery(query=query.text))
    return _stamp(_records_for(recorder, query.query_id), query, routes.routes.get(query.query_id), app_git_sha)


def run_conversation(coordinator: CoordinatorAgent, conv: Conversation, recorder: Recorder,
                     routes: RouteCapture, app_git_sha: str | None = None) -> list[CallRecord]:
    history: list[tuple[str, str]] = []
    out: list[CallRecord] = []
    for turn_index, turn in enumerate(conv.turns):
        QUERY_CONTEXT.set(QueryContext(turn.query_id, conv.conversation_id, turn_index))
        window = history[-HISTORY_WINDOW:]                     # load_history returns the last HISTORY_WINDOW rows
        resolved = condense_query(window, turn.text)           # no-op on turn 0 (empty history)
        result = coordinator.handle_query(UserQuery(query=resolved, raw_query=turn.text))
        history.append(("user", turn.text))                    # save_exchange persists the RAW query
        history.append(("assistant", result["response"]))
        out.extend(_stamp(_records_for(recorder, turn.query_id), turn, routes.routes.get(turn.query_id), app_git_sha))
    return out


def read_raw(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _done_ids(path: Path) -> set[str]:
    done = set()
    for r in read_raw(path):
        done.add(r["conversation_id"] or r["query_id"])
    return done


def capture_all(queries: list[GeneratedQuery], conversations: list[Conversation], out_path: Path,
                workers: int = 4, app_git_sha: str | None = None) -> tuple[int, int]:
    """Run everything not already in out_path; append records as each unit
    finishes so an interrupted capture resumes where it stopped.

    Returns (records written, units that failed). Each unit's records are
    written as a single buffered write so an interrupt or OSError mid-unit
    never leaves a partial unit on disk. A unit whose call raises (an
    APIError after the SDK's retries, a Chroma error, ...) is skipped and
    counted as a failure instead of aborting the whole run; it stays absent
    from out_path so the next invocation re-runs it."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)
    pending_q = [q for q in queries if q.query_id not in done]
    pending_c = [c for c in conversations if c.conversation_id not in done]
    coordinator = CoordinatorAgent()
    written = 0
    failures = 0
    with Recorder() as recorder, RouteCapture.installed() as routes, out_path.open("a", encoding="utf-8") as fh:
        def unit_q(q):
            return run_single_turn(coordinator, q, recorder, routes, app_git_sha)

        def unit_c(c):
            return run_conversation(coordinator, c, recorder, routes, app_git_sha)

        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            unit_of = {}
            futures = []
            for q in pending_q:
                fut = pool.submit(unit_q, q)
                unit_of[fut] = q.query_id
                futures.append(fut)
            for c in pending_c:
                fut = pool.submit(unit_c, c)
                unit_of[fut] = c.conversation_id
                futures.append(fut)
            for fut in as_completed(futures):
                try:
                    recs = fut.result()
                except Exception:
                    failures += 1
                    print(f"capture: unit {unit_of[fut]} failed; it will be re-run on resume", file=sys.stderr)
                    traceback.print_exc()
                    continue
                fh.write("".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in recs))
                fh.flush()
                written += len(recs)
        finally:
            # On Ctrl+C: drop everything still queued so workers stop after their
            # current unit instead of draining the whole run; finished units were
            # already written, so the next invocation resumes from there.
            pool.shutdown(wait=True, cancel_futures=True)
    return written, failures
