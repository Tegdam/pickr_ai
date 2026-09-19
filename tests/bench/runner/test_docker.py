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


def test_image_digest_falls_back_to_id_when_repo_digests_empty(monkeypatch):
    """Locally built images (e.g. bench-client) carry no RepoDigests."""
    calls = []

    def fake_subprocess_run(cmd, check, capture_output, text):
        calls.append(list(cmd))
        if cmd[-1] == "{{index .RepoDigests 0}}":
            return SimpleNamespace(stdout="\n", stderr="")
        assert cmd[-1] == "{{.Id}}"
        return SimpleNamespace(stdout="sha256:local-id\n", stderr="")

    import bench.runner.docker as docker_mod

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_subprocess_run)

    d = Docker()
    assert d.image_digest("bench-client:v0.29.0") == "sha256:local-id"
    assert len(calls) == 2


def test_image_digest_uses_repo_digests_when_present(monkeypatch):
    def fake_subprocess_run(cmd, check, capture_output, text):
        return SimpleNamespace(stdout="vllm/vllm-openai@sha256:abc\n", stderr="")

    import bench.runner.docker as docker_mod

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_subprocess_run)

    d = Docker()
    assert d.image_digest("vllm/vllm-openai:v0.29.0") == "vllm/vllm-openai@sha256:abc"
    assert len(d.command_lines) == 1


def test_ps_names_lists_running_container_names(monkeypatch):
    def fake_subprocess_run(cmd, check, capture_output, text):
        return SimpleNamespace(stdout="bench-r1\nbench-r1-client\n", stderr="")

    import bench.runner.docker as docker_mod

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_subprocess_run)

    d = Docker()
    assert d.ps_names() == ["bench-r1", "bench-r1-client"]
    assert d.command_lines == [["docker", "ps", "--format", "{{.Names}}"]]


def test_ps_names_empty_when_nothing_running(monkeypatch):
    def fake_subprocess_run(cmd, check, capture_output, text):
        return SimpleNamespace(stdout="\n", stderr="")

    import bench.runner.docker as docker_mod

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_subprocess_run)

    d = Docker()
    assert d.ps_names() == []


def test_container_logs_is_one_merged_chronological_call(monkeypatch):
    calls = []

    def fake_subprocess_run(cmd, stdout, stderr, text):
        calls.append((list(cmd), stdout, stderr, text))
        return SimpleNamespace(stdout="line1\nline2\n")

    import bench.runner.docker as docker_mod

    monkeypatch.setattr(docker_mod.subprocess, "run", fake_subprocess_run)

    d = Docker()
    logs = d.container_logs("name")
    assert logs == "line1\nline2\n"
    assert len(calls) == 1
    cmd, stdout, stderr, text = calls[0]
    assert cmd == ["docker", "logs", "name"]
    assert stdout is docker_mod.subprocess.PIPE
    assert stderr is docker_mod.subprocess.STDOUT
    assert text is True
