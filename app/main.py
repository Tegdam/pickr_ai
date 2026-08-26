import logging

from dotenv import load_dotenv

load_dotenv()  # Must run before importing .api/.agents, which read OPENAI_API_KEY at import time.

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router
from .conversation import init_db

init_db()

app = FastAPI(title="Pickr AI")
app.include_router(router, prefix="/api")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
