# Decision Log

A chronological record of the decisions made while building Pickr AI (formerly
SmartShop AI), the reasoning behind them, the alternatives that were
considered and set aside, problems that came up along the way and how they
were resolved, and the handful of cases where an earlier decision was
reversed. `DECISIONS.md` covers the fine-grained *code-design* rationale for
`app/agents.py` and its neighbors in more technical depth; this log gives the
project's own narrative, including process and scope decisions that never
lived in `DECISIONS.md` at all.

Where a decision was later reversed, it's marked **Reverted** with what
changed and why.

---

## 1. Core architecture

**Decision:** A `CoordinatorAgent` routes each query to one of six
specialized agents (review summarization, product recommendation, product
comparison, price comparison, store policy, FAQ) via fixed-priority keyword
matching, rather than an LLM-based intent classifier.

**Why:** Keyword matching is free, deterministic, and easy to test. Several
branches would otherwise overlap on the same query (e.g. "compare the price
of X and Y" matches both "compare" and "price"), so priority order matters:
review → price-comparison → compare → policy → recommendation (default).

**Alternative rejected, then partially revisited:** a full LLM-based router
was considered again as "Path to A+" item 7 and explicitly skipped at that
time — keyword routing had held up well enough, and a full replacement
risked both non-deterministic behavior across the existing routing test
suite and a new LLM call on every query, including the two agents that make
zero OpenAI calls today. Revisited once more afterward with a narrower
option: rather than replacing keyword routing, a single LLM classifier call
was added as a fallback that only runs when no keyword rule matches at all
— catching the specific novel-phrasing failure mode without touching the
tested deterministic paths or adding cost to them. See
`CoordinatorAgent._classify_intent` and `DECISIONS.md`'s "Revisited: LLM
fallback for queries with no keyword match" for the full account.

**PriceComparisonAgent → ProductRecommendationAgent fallback:** Originally,
if `PriceComparisonAgent` couldn't find two named products it returned a dead
end ("please mention two products"). Live testing of chat-history follow-ups
(see §4) surfaced that a query like "what about something cheaper" condenses
into something that still says "cheaper" but names zero or one product — it's
a relative recommendation request, not a comparison. Fixed by falling through
to `ProductRecommendationAgent`, mirroring the existing StorePolicy → FAQ
fallback shape. **Known limitation, accepted:** the fallback doesn't carry
over an explicit price ceiling from the prior turn, so it can occasionally
resurface the same product rather than a cheaper one — deferred.

## 2. FAQ retrieval: full-context → RAG — **Reverted**

**Initial decision:** Pass the entire store-policy list (19 rows, ~2-3KB) as
LLM context on every call. RAG was assessed as unnecessary complexity at that
scale — no indexing pipeline, no new heavy dependency needed.

**Reverted to:** RAG (OpenAI `text-embedding-3-small` embeddings + ChromaDB),
implemented anyway at the user's explicit request — both as a learning
exercise and because the project is meant to eventually run as a product
where the policy set could grow past what fits in one prompt.

**Alternatives considered and rejected:**
- Local embeddings via `sentence-transformers` — rejected, adds a large
  `torch` dependency and a model download for no benefit at this scale.
- Hand-rolled numpy cosine-similarity index — simpler and dependency-free,
  but ChromaDB was preferred as the more "real" building block.
- In-memory ephemeral Chroma client, rebuilt every startup — simpler, but
  re-pays the embedding API cost on every process boot. Persistent client
  (`data/chroma_db`, rebuilt only when the index is stale) chosen instead.

**Sub-decision, also reverted — staleness detection:** Originally the index
rebuilt only when `collection.count() != len(policies)`, which misses an
in-place edit to an existing policy row (e.g. a changed return window) since
the row count doesn't change. **Fixed** by switching to a content hash
(`sha256` over sorted chunk text, stored in the collection's own metadata) —
any addition, removal, *or* edit now triggers a full rebuild. Covered by a
test that asserts the embedding call count directly (1 call on first build,
still 1 on an unchanged rebuild, 2 after an edited-but-same-count policy
list).

## 3. Observability: logging and tracing

**Decision:** `CoordinatorAgent.handle_query` emits one structured `logfmt`
log line per query (`agent`, `status`, `elapsed_ms`, `query`) from a single
`try/finally` around the whole dispatch, rather than a log call duplicated in
every branch or `extra=`-based structured fields (which would need a custom
root formatter affecting every other logger in the process).

**Decision:** Switched off the bare `openai` module-level singleton onto an
explicit `client = wrap_openai(OpenAI(...))` instance, since LangSmith's
`wrap_openai` patches a client instance, not the module proxy. With
`LANGSMITH_TRACING` unset it's a no-op passthrough — shipping this required
no LangSmith account.

**Later addition (this project's most recent commit before this log):**
`wrap_openai` traces each individual OpenAI call, but with no shared parent,
a single query's 1–6 OpenAI calls (condensation, input guardrail, generation,
output guardrail, etc.) each showed up as a disconnected trace in the
LangSmith UI. Added `@traceable(name="handle_conversational_query")` around
the orchestration function in `app/conversation.py` so all of a query's calls
now nest under one parent trace. Also a no-op when tracing is disabled.

**Client construction was later split out** of `app/agents.py` into
`app/openai_client.py` specifically so `app/guardrails.py` could share the
same client without a circular import (`agents.py` calls into
`guardrails.py`; `guardrails.py` can't import the client back out of
`agents.py`). Existing tests that patched `agents.client.chat` kept working
unchanged, since they patch attributes on the shared object, not the
module-level name.

## 4. Chat history (`app/conversation.py`)

**Decision:** Persisted to RDS MySQL (already provisioned) rather than
client-carried history (no server storage — simplest, considered first) or a
from-scratch session store, since a later persona feature would likely need
real persistence anyway.

**Decision:** `CoordinatorAgent` and all six specialized agents stay
completely unaware history exists. One orchestration function,
`handle_conversational_query`, owns everything: load history → condense the
raw query against it → route the resolved, self-contained query through
`CoordinatorAgent.handle_query` unchanged → persist the exchange. Follow-ups
are resolved via a single condensation LLM call rather than passing raw
history into every agent's own prompt — this is also the only approach that
fixes reference resolution ("it", an omitted product name) for the
*deterministic* agents that never call an LLM at all.

**Decision:** `check_input` (prompt-injection detection) must run on the
*raw* incoming message, not the condensed one, so it always sees literal
user input rather than an LLM-rewritten version — enforced via an optional
`raw_query` field on `UserQuery` that the guardrail prefers when present.

**Problem hit — this was documented as done but wasn't, caught while
writing this report:** `handle_conversational_query` was passing only the
condensed query into `CoordinatorAgent.handle_query`, so on every follow-up
turn the injection guardrail was actually checking an LLM-rewritten version
of the message, not what the customer typed — the opposite of the decision
above. Found by re-tracing the code path to describe it accurately here,
not by a test failure. **Fixed** the same day via the `raw_query` field
rather than a second `check_input` call, to avoid doubling the guardrail's
OpenAI/moderation call count per query.

**Problem hit:** a `BigInteger` primary key on `chat_turns` broke the
in-memory SQLite tests, since `BigInteger` doesn't get SQLite's
autoincrement-rowid special-casing. **Fixed** by using a plain `Integer` —
standard for this scale on MySQL anyway, so not a real tradeoff.

**Decision:** RDS connection secured with TLS + certificate/hostname
verification (`ssl_verify_cert` + `ssl_verify_identity` against AWS's public
CA bundle), added specifically because chat history was new infrastructure
touching real customer conversation data.

**Fails open on any DB error** (unreachable host, bad credentials, table not
yet created), same philosophy as guardrails: losing chat history isn't a
safety issue, so a DB hiccup degrades a turn to "no history" instead of
failing the request.

## 5. Input/output guardrails

**Decision:** `check_input` (prompt-injection + off-topic detection, one
combined LLM classifier call, plus OpenAI Moderation) runs before routing;
`check_output` (hallucination/faithfulness classifier + Moderation) runs
inside each of the four LLM-generating agents against whatever context that
agent already built.

**Alternative rejected:** regex-based injection/off-topic detection — too
easy to bypass by rephrasing, and too unreliable for open-ended topicality.
**Alternative rejected:** folding moderation into the same classifier prompt
as a third JSON field — would cut the call count from 5 to 3 per query, but
a purpose-built moderation model was judged measurably more reliable than
asking a general chat model to self-judge harm in the same breath as
injection/hallucination.

**Decision:** Fail-open on infrastructure errors (a broken classifier call
doesn't take down `/api/query`), fail-closed on detected violations (a
successful detection always replaces the response, no bypass).

**Decision:** Block messages for injection/moderation hits are deliberately
generic and don't reveal which check tripped, to avoid coaching an attacker
toward a bypass. Off-topic queries get a more helpful, distinct message,
since that's a benign steering case. The specific reason is still logged.

### Problem: false-positive fallback messages, root-caused and fixed

**Problem hit:** the user noticed some real queries were returning the
generic fallback message. Investigation (via real RDS conversation history,
not synthetic tests) found the guardrail's hallucination classifier — not
`CoordinatorAgent` routing, which was the first suspect — was the cause. Two
concrete root causes were found:

1. The classifier's own system prompt stated an exception ("a response that
   says it can't find an answer is NOT a hallucination") but didn't reliably
   honor it even at `temperature=0`. **Fixed** by adding four concrete
   few-shot examples built directly from the real observed failures (two
   honest hedges → `is_hallucination: false`, two genuine fabrications →
   `is_hallucination: true`). Verified: previously-blocked honest hedges now
   pass 3/3, genuine fabrications still blocked 3/3.
2. The UI's example placeholder text referenced a product ("the Alpha
   Laptop") that doesn't exist in `data/products.csv` — any query using it
   verbatim produced a genuine fabrication the guardrail correctly caught.
   **Fixed** by swapping the placeholder to reference a real, in-stock,
   top-rated product ("the Pro Book v60930"), confirmed via live testing to
   actually appear in its own category shortlist.

**Explicit constraint honored:** the user was clear that `data/*.csv` must
never be modified, since it was provided as fixed project data — so the fix
was to correct the UI's placeholder copy, not to add a fabricated product
into the catalog to make the placeholder "true."

## 6. CSV data: cleaning and in-process caching

**Decision:** `app/data_cleaning.py` cleans each catalog DataFrame with
pandas (`drop_duplicates`, `dropna` on essential columns including
whitespace-only fields, `.clip()` on numeric ranges, `pd.to_datetime` with
`errors="coerce"`) right after `pd.read_csv`, before rows become Pydantic
models. `data/*.csv` turned out to already be clean end to end (row counts
unchanged) — the cleaning stays in place as a correctness guarantee for
anyone hand-editing the CSVs later, and was verified separately against
synthetic dirty rows.

**Discussed before implementing:** whether cleaning at load time would scale
as table sizes grow, versus cleaning offline into pre-cleaned CSVs.
**Decision: keep cleaning at load time.** Vectorized pandas calls are cheap
even at low tens-of-thousands of rows, and an offline step adds a build stage
that can silently drift out of sync if the raw CSV is edited without
rerunning it. The actual performance problem turned out to be elsewhere
(below), and fixing that made "clean at load time" fully fine to keep.

**Decision, approved after the above discussion:** cache each `load_*()`
function's result in-process via `functools.lru_cache(maxsize=1)`, returning
a `tuple` instead of a `list`. This addressed a standing performance TODO:
every specialized agent calls `load_products()`/etc. in its own `__init__`,
and agents are re-instantiated on *every request* by `CoordinatorAgent`, so
without caching each query re-read, re-parsed, and re-cleaned the CSVs from
disk. `tuple` instead of `list` was a deliberate pairing — since the cached
object is now shared across every request, an accidental in-place mutation
would silently corrupt state for every other request; a tuple fails loudly
instead. This decision was also logged and explained in `DECISIONS.md` at
the user's request, alongside its rationale.

**Alternative rejected:** moving load+cache to app startup and injecting data
into agents via constructor arguments. Rejected as a larger refactor than the
problem needed — `lru_cache` gets the same effective result by changing three
functions in `db.py`, with zero changes to `agents.py`.

## 7. Reliability

**Decision:** `app/openai_client.py` sets `max_retries=3` explicitly (up from
the SDK default of 2), rather than hand-rolling retry/backoff with
`tenacity`. The SDK already retries retryable errors with exponential
backoff; wrapping every call site externally would mean either double
retries or disabling the SDK's own retry to avoid that.

**Decision:** `api.py`'s exception handler no longer returns `str(e)` as the
HTTP `detail` — replaced with a fixed `GENERIC_ERROR_MESSAGE`, with the real
exception now logged server-side via `logger.exception(...)`. The raw
exception text could carry internal details (stack traces, DB connection
info, file paths) that shouldn't reach the client. The existing test
asserting on the leaked message was updated to assert the generic message
and the absence of the raw exception text instead.

## 8. Testing and CI

**Decision:** Tests use isolated in-memory fixtures (monkeypatched
`load_products`/etc.) rather than the real CSVs, so tests stay correct
regardless of data content. OpenAI calls are mocked via
`unittest.mock.MagicMock`; no test needs a real API key. An autouse
`bypass_guardrails` fixture keeps pre-existing call-count assertions from
breaking once guardrails added their own calls to the same mocked client,
with guardrail wiring itself tested explicitly and separately.

**Problem hit:** `chromadb.Client()` (in-memory) caches its underlying system
across calls with identical default settings within one process, so two
tests naively creating "fresh" clients leaked data through a shared
collection. **Fixed** by explicitly deleting the collection before each test
that uses the fixture.

**Decision:** Added a minimal GitHub Actions workflow
(`.github/workflows/tests.yml`) — `pip install -r requirements.txt` +
`pytest` on every push/PR. No lint step, no Python version matrix, no
coverage gate: a minimal workflow that actually runs was judged better than
an elaborate one that's more to maintain for a one-maintainer project.
Verified the suite passes from a zero-secrets fresh clone before adding this,
since that's exactly the environment Actions runs in.

**Decision:** Extended the existing `ragas` eval pattern (correctness,
faithfulness, relevancy via `SingleTurnSample`) from FAQAgent-only coverage
to `ProductRecommendationAgent` and `ProductComparisonAgent`. This required
extracting `shortlist_context`/`matched_products`/`product_context` out of
the agents as reusable methods, so eval scripts retrieve the *exact* context
the LLM call actually saw rather than re-deriving matching logic that could
drift out of sync. Eval datasets were built by querying the real product
catalog live (not fabricated), respecting the same "never modify
`data/*.csv`" constraint from §5.

**Result, not just implementation — informative finding logged:**
`ProductComparisonAgent` scored perfect faithfulness (1.00); it sticks to
literal catalog facts. `ProductRecommendationAgent` scored lower (0.41 avg)
because it elaborates with interpretive phrasing ragas' metric doesn't count
as directly supported by context, even though it isn't a fabrication —
consistent with the same paraphrase-vs-verbatim strictness pattern found in
the output guardrail investigation (§5). Logged as a known gap, not acted on.

## 9. Dependency cleanup

**Decision:** Removed `motor`, `streamlit`, and bare `langchain` from
`requirements.txt` — confirmed via repo-wide grep that none were imported
anywhere (leftovers from a direction never built; the project uses
SQLAlchemy/PyMySQL and a static HTML frontend instead). Verified with a
fresh venv + full test suite after removal.

## 10. Code reuse: agents.py

**Decision:** Extracted the copy-pasted `output_check = check_output(...)`
pattern (duplicated across four agents) into one module-level
`_guarded_response()` helper. Pure extraction, verified behavior-preserving
via an unchanged test pass count rather than only by inspection. Done
alongside the eval-coverage extraction in §8, since both needed the same
underlying context-building methods pulled out of the agent methods.

## 11. Project scope: "Path to A+" review

**Decision:** After a from-scratch, evidence-based comparison against the
project's grading criteria, a prioritized punch list of 7 gaps was produced.
The user chose to implement items 1 (remove dead deps), 2 (DRY refactor via
the guardrail helper), 3 (CI workflow), 5 (retries + stop leaking
exceptions), and 6 (extend eval coverage) — captured above in §6–§10.

**Explicitly held, not implemented:** item 4, deploying to Hugging Face
Spaces — the user wanted to hold this because it might need more supporting
implementation first; tracked as a to-do for when it's actually set up
(README's Deployment section notes the Dockerfile targets HF Spaces but
isn't live yet).

**Explicitly skipped:** item 7, redesigning routing from keyword matching to
an LLM-based classifier (see §1) — a deliberate scope decision, not an
oversight.

## 12. Documentation: README

**Discussed before drafting, at the user's request** ("let's talk about this
first, I am not sure what a good README looks like"): what a good README
looks like for this project — simple, appealing, with an architecture
illustration (specifically agent routing, not full system detail), respecting
standard practice, and uncertain whether to reference the not-yet-live
Hugging Face Spaces deployment. Resolved: Mermaid flowchart of just the
`CoordinatorAgent` routing (both fallback chains included), and HF Spaces
left off the README body with a to-do to add a demo link once it's actually
deployed.

**Revised, in a later pass:** the run instructions were first written
assuming dependencies were already installed. **Reverted/rewritten** at the
user's explicit correction: "do not assume all requirements are installed
... write it from the point of view of someone seeing this repo on GitHub
and trying to run it just to see what it is about." Rewritten for a cold
clone: git clone → venv → `pip install -r requirements.txt` → `.env` with
only `OPENAI_API_KEY` required, DB_* vars noted as optional.

**Decision:** `DECISIONS.md` is the user's personal working notes, not meant
to stay in `main` long-term — never linked or referenced from the README or
any other public-facing doc, even though it remains tracked in git for now.

## 13. Git history and attribution — **Reverted**

**Initial decision:** Stop adding the `Co-Authored-By`/`Claude-Session`
commit trailer, for future commits only — existing history left as-is. Chosen
via an explicit either/or question, with "future commits only" as the
recommended and selected option.

**Reverted to:** After the user reported still seeing Claude listed as a
GitHub contributor (existing history still carried the trailer), asked again
and explicitly chose full history rewrite + force-push. Executed via `git
filter-branch --msg-filter` with a `sed` strip of the trailer lines, across
every affected commit on both `performance` and `main`.

**Verification, not just execution:** every rewritten commit was checked
pairwise against its original for identical tree hash, author, and commit
date before the force-push, and original commit objects were confirmed still
reachable via `git cat-file -t` before considering the rewrite safe. After
pushing, GitHub's contributor list was re-checked directly (including
fetching the URL the user provided) to confirm Claude no longer appeared.

**Recurrence (2026-09-03) — GitHub-side cache, not a git problem:** the user
again reported seeing `claude` listed as a contributor, this time on the repo
homepage's right-sidebar "Contributors" widget. Before touching git at all,
re-verified from scratch via three independent sources: (1) `git log --all`
across every commit/branch/tag — zero trace of Claude/Anthropic anywhere;
(2) GitHub's `/repos/.../contributors` REST API — returns only `Tegdam`, 57
contributions; (3) GitHub's dedicated `/graphs/contributors` page, visited
live — it explicitly recomputed ("Crunching the latest data…") and then also
showed only `Tegdam`. Only the repo homepage's sidebar widget specifically
still showed the stale `claude` entry, even immediately after the graphs
page's recompute. Conclusion: the underlying git history has been clean
since the original rewrite above — this is a GitHub-side caching artifact
isolated to that one homepage widget, which lags independently of the API
and the graphs page and has no user-triggerable refresh. **Do not repeat the
`git filter-branch` rewrite for this** — there is nothing left in reachable
history to strip, so a rewrite would be pure risk (another force-push) with
no possible effect on a display cache. If reported again: re-verify via the
same three sources first: `git log --all | grep -i claude`, the contributors
API, and `/graphs/contributors` (visiting it forces a recompute). If all
three are clean, the fix is to wait for GitHub's sidebar cache to expire on
its own (observed to take at least until 2026-09-03 from whenever the prior
rewrite happened) — or GitHub Support if it persists unreasonably long.

## 14. Frontend evolution

**Decision:** Price-tag/receipt-printer visual concept (query styled as a
physical price tag with a punch hole; answers "print" onto a torn-edge
receipt strip) instead of a generic chat-bubble UI, grounded in the retail
domain. Plain HTML/CSS/JS, no framework or build step.

**Fixed during implementation — a small local revert:** the query card's
corner notch was first attempted with a CSS `clip-path` polygon, but the
border didn't render along the cut diagonal edge and it overlapped the
punch-hole element oddly. **Reverted** to a plain punch-hole circle on the
card's left edge, which renders correctly in every browser with no
clip-path artifacts — caught by loading the page in a real browser rather
than only reading the CSS.

**Iterative UI requests, each implemented and then built on** (renamed to
"Pickr"; query moved above response with no divider; new entries append
instead of replacing, clearing the input field afterward; a sessions sidebar
added backed by RDS conversation history; a torn-paper frame added around
the receipt; a conversation header added with left-justified CONV ID and
right-justified DATE/TIME, key-value styled; a footer added with "CUSTOMER
COPY" and a thank-you line). Each round was broken into its own logical git
commit at the user's request, rather than one large commit, then pushed and
merged into `main`.

**Decision:** Typewriter reveal is skipped under `prefers-reduced-motion`;
focus states are visible; the response region uses `aria-live="polite"`.

## 15. Deployment

**Decision:** `python:3.11-slim` base image running `uvicorn` on port 7860,
matching Hugging Face Spaces' Docker SDK default port — set up in advance of
actually deploying (see §11's held item 4).

**Decision:** Added `.dockerignore` (missing from the initial Dockerfile) to
keep `.env` and the local venv out of the build context — without it, `COPY
. .` would have baked the API key directly into an image layer. The key is
expected to be injected at runtime instead. Verified with a local
`docker build` + `docker run`.

## 16. Parked and deferred (not implemented this session)

- **Sentiment classifier for review summarization:** a supervised TF-IDF +
  Logistic Regression classifier over `data/reviews.csv`, proposed as the
  project's one deliberate traditional-ML component, with a known class
  imbalance wrinkle (very few 1–2.5-rated reviews). Explicitly parked pending
  further thought — not rejected, just not yet designed.
- **Persona/personalization:** referenced as a reason to prefer RDS-backed
  chat history over throwaway session storage, but not itself built.
- **OPENAI_API_KEY rotation:** flagged earlier in the engagement as
  outstanding, still not actioned.
- **Inference-serving benchmark sub-project**
  (`smartshop-inference-benchmark-brief.md`, vLLM/SGLang, speculative
  decoding): a separate, explicitly out-of-scope-for-now proposal. Left
  untracked in git throughout, per the user's standing decision.

---

## Summary: decisions later reversed

| Decision | Reversed to | Why |
|---|---|---|
| FAQ retrieval: pass full policy list as context | Implemented RAG (embeddings + ChromaDB) | Learning exercise + anticipated growth past one-prompt scale |
| RAG staleness check: row-count comparison | Content-hash comparison | Row-count missed in-place edits to existing policy rows |
| Co-Authored-By trailer: strip for future commits only | Rewrote and force-pushed full git history to strip it everywhere | Trailer was still visible on GitHub as a contributor after the first fix |
| Query card corner: CSS `clip-path` notch | Plain punch-hole circle | Clip-path broke border rendering and overlapped the punch hole |
| README run instructions: assume deps installed | Rewritten for a cold GitHub clone with no assumptions | User wanted a first-time visitor to be able to run it unaided |
