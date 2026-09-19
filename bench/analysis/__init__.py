"""P0b calibration analysis: pure reducers over the runner's on-disk
artifacts (config.yaml/summary.json/requests.jsonl). This package never
imports `bench.runner` -- it is meant to run standalone (e.g. from a report
notebook) against results a sweep already produced. See `reducers.py`.
"""
