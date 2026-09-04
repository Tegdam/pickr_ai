# FastAPI entrypoint. The statement order in this module is load-bearing --
# see the comments below before rearranging it.

import logging

from dotenv import load_dotenv

load_dotenv()  # Must run before importing .api/.agents, which read OPENAI_API_KEY at import time.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# Imported after load_dotenv/basicConfig on purpose (hence not at the top of
# the file): these modules read environment variables and create loggers as a
# side effect of being imported.
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router
from .conversation import init_db

# Creates the chat_turns table if it's absent. Safe to call on every start,
# and logs-and-continues if the database is unreachable, so an RDS outage
# costs conversation history rather than the whole app.
init_db()

app = FastAPI(title="Pickr AI")
app.include_router(router, prefix="/api")
# Mounted last and at "/", so it acts as the catch-all: every route not
# claimed by the API router above falls through to the static frontend.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
