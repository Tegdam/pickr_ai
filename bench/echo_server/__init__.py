"""OpenAI-compatible echo server (Task 9): a local-process stand-in for a real
inference engine, used to measure the client/harness's own overhead ceiling.

See `bench/echo_server/server.py` for the aiohttp app and
`bench/runner/engine.py`'s `ENGINES["echo"]` for how the runner launches it.
"""
