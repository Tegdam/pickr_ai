"""Docker is a thin subprocess wrapper; only exercised through FakeDocker in
lifecycle tests, plus this one smoke assertion that its command lines are
well-formed (no real docker is ever invoked in tests)."""
from types import SimpleNamespace

from bench.runner.docker import Docker


def test_run_builds_well_formed_command_line(monkeypatch):
    captured = {}

    def fake_subprocess_run(cmd, check, capture_output, text):
        captured["cmd"] = list(cmd)
        captured["check"] = check
        return SimpleNamespace(stdout="cid123\n", stderr="")

    import bench.runner.docker as docker_mod

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_subprocess_run)

    d = Docker()
    cid = d.run(
        "image", "name", ["arg1", "arg2"],
        gpus=True, network_host=True,
        mounts=[("a", "b", "ro")], env={"K": "V"}, entrypoint="x",
        extra_args=["--shm-size", "2g"],
    )

    assert cid == "cid123"
    assert captured["cmd"] == [
        "docker", "run", "-d", "--name", "name", "--ipc=host",
        "--gpus", "all", "--network", "host",
        "-v", "a:b:ro",
        "-e", "K=V",
        "--shm-size", "2g",
        "--entrypoint", "x",
        "image", "arg1", "arg2",
    ]
    assert d.command_lines == [captured["cmd"]]


def test_run_without_gpus_network_or_extras_omits_those_flags(monkeypatch):
    def fake_subprocess_run(cmd, check, capture_output, text):
        return SimpleNamespace(stdout="cid\n", stderr="")

    import bench.runner.docker as docker_mod

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_subprocess_run)

    d = Docker()
    d.run("image", "name", [], gpus=False, network_host=False)
    cmd = d.command_lines[-1]
    assert "--gpus" not in cmd
    assert "--network" not in cmd
    assert cmd == ["docker", "run", "-d", "--name", "name", "--ipc=host", "image"]
