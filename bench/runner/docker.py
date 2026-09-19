"""Thin docker CLI wrapper. Every method maps to one command line; nothing here
knows about engines -- EngineSpec.env / .docker_extra_args are the caller's job
to pass in (see bench/runner/engine.py)."""
from __future__ import annotations

import subprocess


class Docker:
    def __init__(self):
        self.command_lines: list[list[str]] = []

    def _run(self, *cmd: str, check: bool = True) -> str:
        self.command_lines.append(list(cmd))
        return subprocess.run(cmd, check=check, capture_output=True, text=True).stdout.strip()

    def run(self, image, name, args, gpus=True, network_host=True, mounts=(), env=None,
            entrypoint=None, extra_args=None) -> str:
        cmd = ["docker", "run", "-d", "--name", name, "--ipc=host"]
        if gpus:
            cmd += ["--gpus", "all"]
        if network_host:
            cmd += ["--network", "host"]
        for host_path, container_path, mode in mounts:
            cmd += ["-v", f"{host_path}:{container_path}:{mode}"]
        for k, v in (env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        if extra_args:
            cmd += list(extra_args)
        if entrypoint:
            cmd += ["--entrypoint", entrypoint]
        return self._run(*cmd, image, *args)

    def stop(self, name: str, timeout: int = 30) -> None:
        self._run("docker", "stop", "-t", str(timeout), name, check=False)
        self._run("docker", "rm", "-f", name, check=False)

    def is_running(self, name: str) -> bool:
        return self._run("docker", "inspect", "-f", "{{.State.Running}}", name, check=False) == "true"

    def image_digest(self, image: str) -> str:
        return self._run("docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}")

    def pip_freeze(self, image: str) -> str:
        return self._run("docker", "run", "--rm", "--entrypoint", "pip", image, "freeze")

    def container_logs(self, name: str) -> str:
        return subprocess.run(["docker", "logs", name], capture_output=True, text=True).stdout + \
               subprocess.run(["docker", "logs", name], capture_output=True, text=True).stderr
