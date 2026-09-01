# Pickr AI

AI shopping assistant — ask about products, reviews, or store policy.

## Run it (demo)

Assumes the virtual environment is already set up with `requirements.txt` installed and `.env` is already populated (`OPENAI_API_KEY`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`).

```bash
# from the repo root, with the venv activated
python -m uvicorn app.main:app --reload
```

(Use `python -m uvicorn`, not bare `uvicorn` — on some setups `uvicorn` resolves to a stray user-level install outside the venv and fails with `ModuleNotFoundError: No module named 'dotenv'`. If that still happens, run `pip install -r requirements.txt` first to make sure everything is installed *inside* the active venv.)

Then open **http://localhost:8000** in a browser and ask a question.

To stop the server: `Ctrl+C`.
