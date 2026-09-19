"""Thin docker CLI wrapper. Every method maps to one command line; nothing here
knows about engines -- EngineSpec.env / .docker_extra_args are the caller's job
to pass in (see bench/runner/engine.py)."""
from __future__ import annotations

import subprocess
from pathlib import Path


class Docker:
    def __init__(self):
        self.command_lines: list[list[str]] = []

    def _run(self, *cmd: str, check: bool = True) -> str:
        self.command_lines.append(list(cmd))
        return subprocess.run(cmd, check=check, capture_output=True, text=True).stdout.strip()

    def run(self, image, name, args, gpus=True, network_host=True, mounts=(), env=None,
            entrypoint=None, extra_args=None) -> str:
        # C1/P28 minor: no --ipc=host -- with it set, docker silently ignores
        # --shm-size (the engines' own docker_extra_args), so /dev/shm stayed
        # at docker's tiny 64 MiB default regardless of the "--shm-size 2g" we
        # thought we were passing.
        cmd = ["docker", "run", "-d", "--name", name]
        if gpus:
            cmd += ["--gpus", "all"]
        if network_host:
            cmd += ["--network", "host"]
        for host_path, container_path, mode in mounts:
            # C1/P28: docker rejects a relative -v source outright ("<path>
            # includes invalid characters for a local volume name") --
            # resolve every host-side mount path to absolute before building
            # the flag, regardless of what the caller passed in.
            host_path = str(Path(host_path).expanduser().resolve())
            cmd += ["-v", f"{host_path}:{container_path}:{mode}"]
        for k, v in (env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        if extra_args:
            cmd += list(extra_args)
        if entrypoint:
            cmd += ["--entrypoint", entrypoint]
        return self._run(*cmd, image, *args)

    def image_present(self, image: str) -> bool:
        """I4/P33: pre-flight checks this before launching -- `docker image
        inspect` exits non-zero (empty stdout) when the image is missing."""
        return bool(self._run("docker", "image", "inspect", image, check=False))

    def stop(self, name: str, timeout: int = 30) -> None:
        self._run("docker", "stop", "-t", str(timeout), name, check=False)
        self._run("docker", "rm", "-f", name, check=False)

    def is_running(self, name: str) -> bool:
        return self._run("docker", "inspect", "-f", "{{.State.Running}}", name, check=False) == "true"

    def ps_names(self) -> list[str]:
        """Names of all currently-running containers (pre-flight check, spec
        §6 step 1: refuse to start while a `bench-*` container is running)."""
        out = self._run("docker", "ps", "--format", "{{.Names}}", check=False)
        return [line for line in out.splitlines() if line]

    def exit_code(self, name: str) -> int:
        out = self._run("docker", "inspect", "-f", "{{.State.ExitCode}}", name, check=False)
        try:
            return int(out)
        except ValueError:
            return -1

    def build(self, tag: str, dockerfile: str, context: str) -> str:
        """Derived-image build (ruling P5, e.g. bench-client:v0.29.0). Never
        invoked against a real docker daemon in tests."""
        return self._run("docker", "build", "-t", tag, "-f", dockerfile, context)

    def image_digest(self, image: str) -> str:
        digest = self._run("docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}", check=False)
        if digest:
            return digest
        # Locally built images (e.g. bench-client) carry no RepoDigests -- fall
        # back to the image Id.
        return self._run("docker", "image", "inspect", image, "--format", "{{.Id}}")

    def pip_freeze(self, image: str) -> str:
        return self._run("docker", "run", "--rm", "--entrypoint", "pip", image, "freeze")

    def container_logs(self, name: str) -> str:
        return subprocess.run(["docker", "logs", name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
