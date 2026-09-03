# Pre-Deploy Checklist

Working list of cleanup items to run through before deploying Pickr AI live.
Not part of the public documentation — personal tracking only.

- [x] **CRLF line-ending drift on several tracked files.** ~~Files edited
  through this session's Windows-side file access get written back with
  CRLF...~~ **Done:** added `.gitattributes` (`* text=auto eol=lf`),
  committed separately. Turned out none of the previously-listed files
  (`static/index.html`, `app/api.py`, `app/data_cleaning.py`, `app/db.py`,
  `app/guardrails.py`, `app/openai_client.py`, `evals/*`, `requirements.txt`,
  `.github/workflows/tests.yml`, `tests/test_api.py`) had ever actually
  been *committed* with CRLF — only the local working tree had drifted, so
  after physically renormalizing them to LF there was nothing left for git
  to commit (`git diff` was already empty). Verified with
  `git ls-files | xargs grep -c $'\r'` — zero hits repo-wide. Going forward,
  `.gitattributes` normalizes automatically on checkout/commit, so this
  shouldn't recur.

- [x] **Rotate `OPENAI_API_KEY`.** ~~The key was exposed in an earlier
  session transcript and has not been rotated since.~~ **Done** — new key
  generated and `.env` updated directly by the user (never pasted into this
  conversation). Verified working via a smoke test (`client.models.list()`,
  164-char key parsed correctly by `dotenv`, authenticated successfully,
  124 models listed) without the key value ever being read or logged by
  the smoke test itself. Still outstanding: if/when deployed live, the new
  key also needs to be set in that platform's own secret store (e.g.
  Hugging Face Spaces secrets) — not yet applicable since not deployed.
  Old key revocation in the OpenAI dashboard is on the user to confirm.

- [x] **Implement explicit conversation deletion.** ~~Add
  `delete_conversation(conversation_id)` to `app/conversation.py`... plus a
  `DELETE /api/sessions/{conversation_id}` route...~~ **Done:**
  `delete_conversation` added (deliberately does *not* fail open — see
  `DECISIONS.md`), `DELETE /api/sessions/{conversation_id}` route added,
  6 new tests (4 in `test_conversation.py`, 2 in `test_api.py`). Report's
  Limitations section updated to describe on-request deletion. Automatic
  time-based expiry is still not implemented — no scheduler exists in the
  app — and remains an open question, not decided either way. No frontend
  button was added (out of scope for this item — the sessions sidebar has
  no delete affordance yet; add if/when wanted).

## Open discussion topics

Not bugs — deliberate scope calls made during development that are worth
revisiting with fresh eyes before going live, not necessarily changing.

- [x] **Discuss: personalized recommendations.** ~~`ProductRecommendationAgent`
  doesn't personalize based on a customer's own history or preferences.~~
  **Resolved: deferred to future work, not implemented.** Checked whether
  the course's original data source might have a personalization-ready
  dataset beyond the three CSVs in `data/` — even if it did, `reviews.csv`
  doesn't identify a reviewer, and Pickr AI has no customer identity system
  at all (`conversationId` is a random UUID per page load, no login). Real
  personalization needs both new data *and* an identity layer, not just a
  CSV with more columns — the identity gap is the actual blocker, and it's
  shared with the price-drop-alerts idea below. Documented as a scope
  boundary in the report's Limitations and Use Cases sections rather than
  built.
- [x] **Discuss: LLM-based query routing.** ~~`CoordinatorAgent` routes on
  fixed-priority keyword matching, not an LLM intent classifier.~~
  **Resolved:** a full LLM router was rejected (regression risk against the
  existing routing tests, a new cost/latency floor on every query including
  the deterministic agents). Implemented instead as a narrow fallback —
  `CoordinatorAgent._classify_intent` only runs when no keyword rule
  matches at all, then dispatches through the same per-category logic. See
  `DECISIONS.md`. Once there's live traffic, `via=llm_fallback` in the
  `coordinator_route` log line shows how often it actually fires.

## Future feature ideas

Net-new extensions, not fixes — from §10 (Use Cases) of the report. Not
scoped or estimated yet. Considered implementing now and decided against it
(see the personalization item above) — not overengineering to *design*
these, but building them now would be, given both share the same
unresolved prerequisite below.

**Shared blocker for both of the below (and for personalization):** Pickr
AI has no customer identity system. `conversationId` is a random UUID
generated fresh per page load, not tied to a login or any persistent
customer record, and no provided dataset identifies an individual customer
either. Any feature that needs to recognize the same customer across
visits needs that identity layer first — it's the actual prerequisite, not
a bigger agent roster.

- [ ] **Cart/checkout agent.** A seventh agent handling adding items to a
  cart and walking through checkout, rather than the assistant stopping at
  recommend/compare/inform. Blocked on customer identity (above); also the
  first genuinely stateful, mutable feature in the app — every existing
  agent is read-only over the catalog.
- [ ] **Price-drop alerts.** Notify a customer when a product they asked
  about drops in price. Blocked on customer identity (above) *and* needs a
  scheduler, which the app has none of today (see the conversation-deletion
  item's note on the same gap).
