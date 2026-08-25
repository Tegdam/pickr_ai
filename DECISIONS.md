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

## Testing

**Decision:** Isolated in-memory fixtures (`monkeypatch`-ed
`load_products`/`load_reviews`/`load_store_policies`) rather than testing
against the real `data/*.csv` files, so tests stay correct and readable
regardless of what's in the CSVs, and can use round numbers for easy
assertions.

**Decision:** `pytest` with `unittest.mock.MagicMock`, mocking
`openai.chat`/`openai.embeddings` at the module level rather than reaching
into the real client objects — `openai.chat`/`openai.embeddings` are lazy
proxies that would otherwise attempt to build a real client (and need
`OPENAI_API_KEY`) the moment an attribute like `.completions` is accessed on
them.

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
