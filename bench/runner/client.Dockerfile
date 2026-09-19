FROM vllm/vllm-openai:v0.29.0
# Pinned to the version in the project's own venv (env/bin/python -c "import
# pandas; print(pandas.__version__)") so the client's pandas is not whatever
# happens to resolve at build time.
RUN pip install --no-cache-dir pandas==3.0.5
