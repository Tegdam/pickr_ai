# Design Decisions

A running log of non-obvious design decisions made while building out the agents
in `app/agents.py`, and why. Code and tests are the source of truth for *what*
exists; this file is for *why* it looks the way it does.

## CoordinatorAgent routing

**Decision:** `CoordinatorAgent.handle_query` checks keywords in a fixed priority
order — `review` → price-comparison trigger → `compare` → policy trigger →
default (recommendations) — because several branches would otherwise overlap
on the same query (e.g. "compare the price of X and Y" contains both "compare"
and "price").

**Alternatives considered:** An LLM-based intent classifier instead of keyword
matching. Rejected for now — keyword matching is free, deterministic, and easy
to test; revisit if query phrasing in practice turns out too varied for
substring checks to keep up with.

## ReviewSummarizationAgent

**Open question (not yet implemented):** the capstone brief calls for
sentiment-based review summarization; today `analyze_reviews` has no
explicit sentiment step — it hands raw review text (with star ratings) to
an LLM summarization prompt and relies on the model to implicitly infer
"praise vs. complaints" sentiment from that.

**Proposal under consideration:** train a supervised sentiment classifier
(TF-IDF + Logistic Regression or Naive Bayes) on `data/reviews.csv` (4,000
rows), deriving labels from the existing `rating` column (e.g. ≥4 positive,
≤2.5 negative, else neutral), and use it to classify each review's sentiment
before the LLM summarizes — real feature engineering / train-eval split /
accuracy-F1 reporting, rather than calling a pretrained lexicon tool (e.g.
VADER), which was considered and set aside as feeling more like a library
call than an ML exercise for this project's goals. Also doubles as the
project's one deliberate traditional-ML component alongside the LLM-heavy
rest of the app.

**Known wrinkle:** `rating` is heavily skewed positive (409 reviews at 5.0,
1353 at 4.0, versus only 6 at 2.0 and 47 at 2.5) — negative is a thin slice
of the data, so class imbalance will need explicit handling (class
weighting, resampling, or just honest reporting of the limitation) rather
than being glossed over.

**Status:** parked — pending further thought before committing to a design.

## ProductRecommendationAgent

**Decision:** Became the coordinator's default fallback (anything not caught
by the other branches), matching what the original stub's placeholder message
already promised.

**Decision:** Structured parsing (category/brand/price-ceiling extracted via
substring/regex matching against the actual catalog's values) feeds a
filtered, ranked shortlist into the LLM call, rather than handing the whole
catalog to the LLM and asking it to pick. Keeps filtering deterministic and
testable, and keeps the LLM from recommending something outside the filtered
set.

**Known limitation:** Category matching requires the query to say the category
name as it's spelled in the catalog (`smart_tv` or "smart tv") — a bare "tv"
won't match. Accepted as out of scope; revisit if real usage shows this is a
common phrasing.

## PriceComparisonAgent

**Decision:** Deterministic (no LLM call) — computes exact `$`/`%` price
deltas between named products, distinct from `ProductComparisonAgent`'s LLM
narrative (which already mentions price but doesn't compute deltas). Routes on
`"cheaper"`, or `"price"` paired with `"compare"/"difference"/"cost"`, checked
*before* the plain `"compare"` branch so it isn't shadowed.

**Decision (added with chat history):** if `PriceComparisonAgent` can't find
two named products (`NOT_ENOUGH_PRODUCTS_MESSAGE`), `CoordinatorAgent` falls
through to `ProductRecommendationAgent` instead of surfacing the "mention two
products" dead end — same fallback shape as `StorePolicyAgent` → `FAQAgent`
below. Found via live testing of query condensation (see "Chat history"
section): a follow-up like "what about something cheaper" condenses into a
standalone question that still contains "cheaper" but names zero or one
product, since there's nothing to compare *against* — it's a relative
recommendation request, not a two-product comparison, even though it hits
the same keyword. This was a pre-existing routing-heuristic edge case
(the same failure happens today for "recommend something cheaper than the
X" typed as a first message, no history involved) that chat history just
made far more likely to surface, since condensed follow-ups naturally reuse
comparison words.

**Known limitation (accepted for now):** the fallback fixes the dead end,
but not full correctness — the condensed query above doesn't carry over an
explicit price number (the condensation model wrote "costs less", not
"under $399"), and `ProductRecommendationAgent._match_price_ceiling` only
extracts constraints from literal "under $X"/"less than $X" phrasing. It
can't reason "cheaper than the specific price I mentioned last turn," so it
falls back to top-rated-in-stock with no price filter, which can resurface
the *same* product rather than a genuinely cheaper one. Fixing this would
need either the condensation prompt to resolve relative price references
into explicit numbers, or `ProductRecommendationAgent` to accept a numeric
anchor from context — deferred as a follow-up, not in scope of chat history
itself.

## StorePolicyAgent / FAQAgent split

**Decision:** `StorePolicyAgent` keeps its original exact-keyword match
(fast, free, no LLM call) as the first attempt. `FAQAgent` is a fallback that
only runs when `StorePolicyAgent` finds nothing — so phrasing like "can I send
this back for a refund" (no literal policy_type word) still gets answered.

## FAQAgent retrieval: RAG vs. full-context

**Initial decision:** Pass the entire policy list (19 rows, ~2-3KB) as LLM
context on every call. At that scale, RAG (embeddings + vector store) was
assessed as unnecessary complexity — no indexing pipeline, no new heavy
dependency, and it trivially fits in a single prompt.

**Revised decision:** Implemented RAG anyway, at the user's request, both as a
learning exercise and because the project is intended to eventually run as a
real product where the policy set may grow well past what fits in one prompt.

- **Embeddings:** OpenAI's `text-embedding-3-small` via the same `openai`
  client already used for chat completions — no new heavy dependency,
  consistent with the rest of the codebase. (Alternative considered: a local
  model via `sentence-transformers` — rejected, adds a large `torch`
  dependency and a model download for no benefit at this scale.)
- **Vector store:** ChromaDB, per explicit user preference. (Alternative
  considered: a hand-rolled numpy cosine-similarity index — simpler and
  dependency-free, but Chroma is the more "real" building block if the
  policy/FAQ set is expected to grow.)
- **Persistence:** `chromadb.PersistentClient` pointed at `data/chroma_db`,
  rebuilt only when `collection.count() != len(policies)`. Avoids re-paying
  the embedding API cost on every process restart. (Alternative considered:
  an in-memory ephemeral client that rebuilds every startup — simpler, but
  pays a real API call on every boot.)
- **Chunking:** one chunk per policy row (`policy_type` + description +
  conditions + timeframe) — the rows are already small, atomic units, so no
  further splitting was needed.

### Staleness detection: count-based → content-hash-based (implemented)

**Original limitation:** the rebuild trigger was `collection.count() !=
len(policies)`, which only notices policies being **added or removed**. If
someone edited a row in place (e.g. changed the return window from 30 to 14
days) without changing the row count, the stale embedding would stay in the
index indefinitely and `FAQAgent` would keep citing the old terms.

**Fix:** `PolicyIndex` now computes `sha256("\n".join(sorted(chunk_texts)))`
on every init and compares it against a `content_hash` stored in the Chroma
collection's own `metadata` (set via `collection.modify(metadata=...)`, read
back via `collection.metadata` — persists across restarts since it lives in
the same Chroma persist directory, no separate manifest file needed). Any
difference — added, removed, *or* edited rows — now triggers `_rebuild_index`,
which does `delete_collection` + recreate rather than a partial upsert. That
also fixed a latent correctness gap in the old upsert-by-index approach: if
the policy count ever shrank, upsert would leave orphaned old-index entries
behind since it only touches the ids it's given.

This is still a blunt "rebuild everything on any change" strategy rather than
incremental re-indexing of just the changed rows — fine at 19 rows; revisit
only if the policy set grows large enough that a full rebuild becomes slow or
noticeably expensive.

Covered by `tests/test_agents.py::TestPolicyIndex::test_rebuilds_only_when_content_changes`,
which asserts the embedding call count directly: 1 call on first build, still
1 after reconstructing with identical content, 2 after reconstructing with an
edited (but same-count) policy list.

## Observability: structured logging + LangSmith tracing

**Decision:** `CoordinatorAgent.handle_query` logs one line per query
(`agent`, `status`, `elapsed_ms`, `query`) via stdlib `logging`, in `logfmt`
style — key=value pairs baked directly into the message string — rather than
passing fields through `extra=`. `extra=` would need a custom root formatter
to render those fields, and that formatter would then apply to every other
logger in the process (uvicorn, httpx, etc.) whose records don't carry them.
Baking the fields into the message sidesteps that while staying greppable.

**Decision:** The log call sits in one `try/finally` wrapped around the whole
routing dispatch, with `agent_name`/`status` tracked as locals updated per
branch, rather than a duplicated log call in each branch. One line always
fires — including on exceptions, where `status` stays `"error"` — without
swallowing the exception itself (it still propagates to `api.py`'s existing
try/except → 500 handler).

**Decision:** Switched `app/agents.py` off the bare `openai` module singleton
(`openai.chat.completions.create(...)`) onto an explicit
`client = wrap_openai(OpenAI(api_key=...))` instance, because LangSmith's
`wrap_openai` patches a client instance, not the module-level proxy. Every
chat/embeddings call in the app is now automatically traced to LangSmith once
tracing env vars are set (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`,
`LANGSMITH_PROJECT`) — with them unset, `wrap_openai` is a no-op passthrough,
so shipping this required no LangSmith account.

**Decision:** Falls back to a placeholder key
(`os.getenv("OPENAI_API_KEY") or "not-set"`) rather than passing `None`
through — `OpenAI()` raises immediately at construction if no key is
resolvable anywhere, which would break test collection (tests monkeypatch
`client.chat`/`client.embeddings` before any real call and never need a real
key to run).

*(Update: `client` construction itself was later moved out to
`app/openai_client.py` — see the Guardrails section below for why.)*

## Guardrails: input/output checks

**Decision:** `app/guardrails.py` runs two checks: `check_input`
(prompt-injection detection, off-topic detection, and OpenAI's Moderation
API) on the raw query before `CoordinatorAgent.handle_query` routes it to
any agent, and `check_output` (a hallucination/faithfulness classifier plus
Moderation) inside each of the four LLM-generating agents
(`ReviewSummarizationAgent`, `ProductRecommendationAgent`,
`ProductComparisonAgent`, `FAQAgent`), checked against whatever context that
agent already built for its own prompt (review text, product shortlist,
matched products, or retrieved policy chunks respectively).
`StorePolicyAgent`'s keyword-hit path and `PriceComparisonAgent` are
untouched — both are fully deterministic with no LLM call, so there's
nothing to hallucinate.

**Decision:** Injection + off-topic detection is one combined LLM classifier
call (JSON response, `{"is_injection": bool, "is_off_topic": bool}`) rather
than two separate calls or regex heuristics. Regex was rejected as too easy
to bypass for injection (trivially defeated by rephrasing) and too
unreliable for open-ended topicality in a shopping domain; combining the two
into one call avoids paying for two round-trips where one already covers
both questions.

**Decision:** Moderation (`client.moderations.create`) runs as its own
separate call on both input and output, in addition to the classifiers,
rather than folding "is this toxic" into the same classifier prompt as a
third JSON field. The dedicated Moderation endpoint uses a purpose-built,
specifically-trained model; asking a general chat model to self-judge harm
categories in the same breath as injection/off-topic/hallucination was
rejected as measurably less reliable, even though it would have cut the call
count from 5 to 3 per query.

**Known cost, accepted deliberately:** an LLM-routed query now makes up to 5
OpenAI calls where it made 1 before — input classifier, input moderation,
the agent's real generation call, output classifier, output moderation.
(`FAQAgent` adds a 6th: its pre-existing retrieval embedding call.)
Correctness/coverage was prioritized over latency/cost for this pass. The
two moderation calls are much cheaper than the two classifier calls (no
token generation, a small dedicated model) — the real cost driver is the
classifier pair, not the raw call count.

**Decision:** Fail-open on infrastructure errors, fail-closed on detected
violations — these are different failure modes. If a classifier or
moderation call itself throws (timeout, malformed JSON, API outage),
`check_input`/`check_output` log a warning and return `blocked: False`, so a
guardrail-service hiccup doesn't take down `/api/query` entirely. If a check
runs successfully and flags something, the response is always replaced — no
bypass on a successful detection.

**Decision:** Block messages deliberately don't reveal *why* for injection
or input-moderation hits — both use the same generic `"I'm not able to help
with that request."` — to avoid coaching an attacker toward a bypass by
telling them which detector tripped. Off-topic queries get a distinct, more
helpful message (`"I can only help with questions about our products,
reviews, and store policies..."`) since that's a benign steering case, not
adversarial. The *specific* reason (`injection` / `off_topic` /
`moderation_input` / `hallucination` / `moderation_output`) is still
recorded in the `coordinator_route` structured log line either way, so
there's no loss of operational visibility.

**Decision:** `app/openai_client.py` was split out of `app/agents.py` (which
previously constructed `client` directly — see the Observability section
above) specifically to support this: `guardrails.py` needs the same
`client`, and `agents.py` needs to call into `guardrails.py`, which would
otherwise be a circular import. Existing tests that do
`monkeypatch.setattr(agents.client, "chat", ...)` kept working unchanged,
since they patch attributes on the shared client object itself, not the
module-level name binding.

## Chat history: app/conversation.py

**Decision:** Persisted to RDS MySQL (already provisioned) rather than
either client-carried history (no server storage, simplest, considered
first) or a from-scratch session store. Using existing infrastructure beat
building throwaway session-scoped plumbing that a later persona feature
would likely need to redo anyway with real persistence.

**Decision:** `CoordinatorAgent` and all six specialized agents in
`agents.py` stay completely unaware history exists. `app/conversation.py`
owns everything conversation-related — loading/persisting turns and
condensing follow-ups — behind one orchestration function,
`handle_conversational_query(conversation_id, raw_query)`, which:
load history → condense the raw query against it (no-op if there's no
history yet) → route the resolved, self-contained query through
`CoordinatorAgent.handle_query` unchanged → persist the exchange → return.
This kept the blast radius of the whole feature to one new module plus a
one-line change each in `api.py`/`main.py`.

**Decision:** Follow-ups are resolved via query condensation — one LLM call
that rewrites e.g. "what about something cheaper" into a standalone
"what's a cheaper alternative to the Alpha Laptop" using the last
`HISTORY_WINDOW` (6) turns — rather than passing raw history into each of
the four LLM-calling agents' own prompts. Condensing once, before routing,
means routing and all six agents work identically whether or not history is
involved, and it's the only approach that also fixes reference resolution
for the *deterministic* agents (`PriceComparisonAgent`,
`StorePolicyAgent`'s keyword path), which don't call an LLM at all and
would otherwise have no way to resolve "it" or an omitted product name.
Skipped entirely when there's no history yet (a conversation's first
message), so no added latency/cost on that turn.

**Decision:** `check_input` (guardrails) still runs on the *raw* incoming
message, before condensation — deliberately, so prompt-injection detection
always sees literal user input rather than a version an LLM has already
rewritten.

**Decision:** `ChatQuery` (`app/models.py`) is a distinct model from
`UserQuery`, not a `conversation_id` field bolted onto `UserQuery` — despite
an earlier decision (see API layer section below) to avoid keeping two
models of the *same* shape. These aren't the same shape: `ChatQuery` is the
API request boundary and needs `conversation_id`; `UserQuery` is what
`CoordinatorAgent`/the six agents consume internally and has no reason to
know about conversations at all. `conversation.py` constructs a fresh
`UserQuery(query=resolved_query)` after condensation, so nothing downstream
of the orchestration function ever sees a `conversation_id`.

**Decision:** Schema is one table, `chat_turns` (`id`, `conversation_id`,
`role`, `content`, `created_at`), created via
`Base.metadata.create_all()` at app startup rather than a migration
framework (Alembic) — no reason to version a single table at this scale,
consistent with how `PolicyIndex` just rebuilds its Chroma collection rather
than migrating it. `id` is a plain auto-incrementing `Integer`, not
`BigInteger` as first drafted — `BigInteger` primary keys don't get
SQLite's autoincrement-rowid special-casing, which broke the in-memory
SQLite tests; `Integer`/`auto_increment` is standard for this scale on
MySQL too, so this wasn't a real tradeoff.

**Decision:** `load_history` orders by `id` (insertion order), not
`created_at`. A user/assistant pair saved in the same `save_exchange` call
can land in the same second, and MySQL's default `DATETIME` resolution is
1-second — ordering by `created_at` alone risked ties putting the assistant
turn before the user turn it was replying to.

**Decision:** `save_exchange` writes both the user message and the final
(post-guardrail) assistant response in one commit, rather than two separate
`save_turn` calls — atomic (both rows land or neither does), and one fewer
round trip per turn on top of the ones chat history already adds.

**Decision:** Fails open on any DB error (unreachable host, bad
credentials, table not yet created) in both `load_history` (returns `[]`)
and `save_exchange` (silently drops the write) — same philosophy as
guardrails' infra-error handling. Losing chat history isn't a safety issue,
so a DB hiccup degrades a turn to "no history" rather than failing
`/api/query` entirely. `init_db()` (called once at startup) follows the
same rule: a failed `create_all()` is logged and swallowed so the app still
starts even if RDS isn't reachable yet.

**Decision:** Connection config is five env vars (`DB_HOST`, `DB_PORT`,
`DB_NAME`, `DB_USER`, `DB_PASSWORD`), assembled into a
`mysql+pymysql://...` URL — not committed anywhere, including this file, to
avoid repeating the earlier incident where a real key ended up printed into
a session transcript.

**Decision:** `conversationId` is a `crypto.randomUUID()` generated once per
page load in `static/index.html` and sent with every request — not
persisted (e.g. via `localStorage`), so a page reload starts a fresh
conversation. Acceptable for a within-conversation feature; persistent
identity across reloads/devices is exactly the kind of thing the deferred
persona feature would need to solve properly (see the parked proposal in
the ReviewSummarizationAgent section above for the same "defer until
actually needed" reasoning).

*(See the PriceComparisonAgent section above for a routing-heuristic bug
this feature's live testing surfaced and fixed, plus a related known
limitation left unfixed.)*

**Decision (testing):** `load_history`/`save_exchange` are tested against a
real in-memory SQLite engine (`tests/test_conversation.py`'s `sqlite_db`
fixture swaps `conversation.SessionLocal`), not mocks — SQLAlchemy makes the
dialect swap trivial, and this exercises real SQL (schema, filtering,
ordering) rather than asserting on a mock's call args. `condense_query` and
the `handle_conversational_query` orchestration *are* mocked at their own
boundaries (LLM client; `load_history`/`condense_query`/`coordinator`/
`save_exchange` respectively), same layered approach as the guardrails
tests. No test needs real RDS credentials to run.

## Testing

**Decision:** Isolated in-memory fixtures (`monkeypatch`-ed
`load_products`/`load_reviews`/`load_store_policies`) rather than testing
against the real `data/*.csv` files, so tests stay correct and readable
regardless of what's in the CSVs, and can use round numbers for easy
assertions.

**Decision:** `pytest` with `unittest.mock.MagicMock`, mocking
`client.chat`/`client.embeddings` (the module-level `wrap_openai(OpenAI(...))`
instance in `agents.py`) rather than issuing real requests — no test needs a
real `OPENAI_API_KEY`, since the module falls back to a placeholder key at
construction time (see Observability section above) and every real call site
is swapped out before it runs.

**Decision:** An autouse `bypass_guardrails` fixture in `test_agents.py`
patches `agents.check_input`/`agents.check_output` to always pass through,
for every test in that file by default. Without it, adding guardrails would
have broken every pre-existing test that asserts an exact LLM call count
(`mock_openai.assert_called_once()` etc.), since `check_input`/`check_output`
add their own calls to the same mocked client — and those tests aren't about
guardrails anyway. Guardrails' own logic is tested in isolation in
`test_guardrails.py`; the wiring (that `CoordinatorAgent` actually calls
`check_input` and honors a block, that each LLM agent actually calls
`check_output` with its own context) is tested explicitly in
`TestCoordinatorGuardrails`/`TestAgentOutputGuardrails`, which override the
autouse bypass per-test via a second `monkeypatch.setattr` call.

**Decision:** `conftest.py` (empty, at repo root) and `pytest.ini` (with
`testpaths = tests`) were added because this project has no packaging/`src`
layout — without them, `from app import agents` wouldn't resolve, and `pytest`
would otherwise try to crawl the entire `env/` virtualenv looking for tests.

**Decision:** RAG tests use hand-crafted deterministic fake embeddings (not
random or hash-based vectors) so that Chroma's real similarity search
produces a predictable ranking — this tests the actual retrieval mechanism
end-to-end (query → embed → Chroma nearest-neighbor → top-k) rather than just
mocking `.query()`'s return value directly.

**Gotcha hit during implementation:** `chromadb.Client()` (in-memory) caches
its underlying system across calls made with identical default settings
within one process — so two tests naively creating "fresh" in-memory clients
still shared the same `"store_policies"` collection and leaked data between
tests. Fixed by explicitly `delete_collection`-ing before each test that uses
the `in_memory_chroma` fixture.

## API layer: api.py / main.py

**Decision:** `api.py`'s `handle_query` takes `app.models.UserQuery` directly
instead of a separate `QueryInput` model — same shape, no reason to keep two.

**Bug fixed:** every agent method already returns `{"response": ...}`, but
`handle_query` re-wrapped that result in another `{"response": ...}`, so
callers received a doubly-nested payload. Fixed by returning the agent's
result directly.

**Decision:** `main.py` (previously empty) is the actual FastAPI entrypoint —
it includes the `api.py` router under `/api`, then mounts `static/` at `/`
via `StaticFiles(html=True)` last, so the mount acts as a catch-all for the
frontend without shadowing `/api/*` (routes registered first win). Dropped
the old `GET /` JSON welcome route from `api.py` since the static frontend
now serves the homepage.

**Decision:** `load_dotenv()` runs in `main.py` before `.api`/`.agents` are
imported, since `agents.py` reads `OPENAI_API_KEY` at import time. Mirrors
the pattern already used in `evals/eval_agents.py`.

## Frontend: static/index.html

**Decision:** Price-tag / receipt-printer visual concept (query card styled
as a physical price tag with a punch hole; answers "print" onto a torn-edge
receipt strip with a typewriter reveal) instead of a generic chat-bubble UI —
grounded in the retail domain the assistant actually serves. Plain HTML/CSS/JS,
no framework or build step, since the page is a single static file served
as-is.

**Decision:** Typewriter reveal is skipped under `prefers-reduced-motion`;
focus states are visible; the response region uses `aria-live="polite"`.

**Fixed during implementation:** the wordmark's icon initially used a
house-icon SVG path pasted by mistake instead of a price tag; corrected to
an actual tag-shaped path with a punched-hole circle. The query card's corner
also originally used a `clip-path` polygon to cut a notch, but the border
didn't render along the cut diagonal edge and the punch-hole element
overlapped it oddly — replaced with a plain punch-hole circle on the card's
left edge, which renders correctly in every browser without clip-path
artifacts. Caught by loading the page in a real browser rather than just
reading the CSS.

## API testing: tests/test_api.py

**Decision:** Tests use `fastapi.testclient.TestClient` against the real
`app`, mocking `app.api.agent.handle_query` directly rather than the
OpenAI/CSV layer underneath it — agent logic is already covered by
`test_agents.py`, so these tests stay focused on the HTTP wrapper: request
validation (422 on a missing `query` field), the success response shape, and
the try/except → 500 path.

## Docker / deployment

**Decision:** `python:3.11-slim` base image running `uvicorn` on port 7860,
matching Hugging Face Spaces' Docker SDK default port.

**Decision:** Added `.dockerignore` (missing from the initial Dockerfile) to
keep `.env` and the `env/` virtualenv out of the build context — without it,
`COPY . .` would have baked `OPENAI_API_KEY` directly into an image layer.
The key is expected to be injected at runtime instead (e.g. a Space secret),
which `os.getenv` already picks up with no code changes. Also excludes
`data/chroma_db/` (the RAG index), rebuilt fresh on first run in the
container.

Verified with a local `docker build` + `docker run`: `.env` and `env/` are
absent from the built image, and `/` and `/api/query` both respond
correctly with the key passed in as a runtime environment variable.
