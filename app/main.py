from dotenv import load_dotenv

load_dotenv()  # Must run before importing .api/.agents, which read OPENAI_API_KEY at import time.

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router

app = FastAPI(title="SmartShop AI")
app.include_router(router, prefix="/api")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
