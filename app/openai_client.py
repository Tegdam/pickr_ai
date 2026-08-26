# Shared OpenAI client, split out from app/agents.py so app/guardrails.py can
# use the same client without a circular import (agents.py calls into
# guardrails.py, so guardrails.py can't import the client back out of agents.py).

import os

from openai import OpenAI
from langsmith.wrappers import wrap_openai

# wrap_openai traces every call this client makes to LangSmith once tracing is
# enabled (LANGSMITH_TRACING=true, plus LANGSMITH_API_KEY/LANGSMITH_PROJECT).
# With those env vars unset it's a no-op passthrough, so wrapping unconditionally
# is safe -- no LangSmith account needed for the app to work.
#
# Fall back to a placeholder key rather than erroring at import time: OpenAI()
# raises immediately if no key is available anywhere, which would break test
# collection (tests monkeypatch client.chat/client.embeddings and never make a
# real call, so no real key is needed for them to run).
client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "not-set"))
