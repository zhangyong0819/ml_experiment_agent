import subprocess
from types import SimpleNamespace
import pytest
from ml_experiment_agent.backends.cluster import ClusterConfig
from ml_experiment_agent.backends.lsf import LSFBackend


def direct(**changes):
    return ClusterConfig(**({"ssh_host": "login.example.edu", "ssh_username": "researcher/10.0.0.1/self",
                            "remote_project_root": "/project/agent"} | changes))


def test_direct_username_is_one_opaque_argument():
    args = direct(ssh_port=2222).ssh_arguments()
    assert args[-5:] == ["-p", "2222", "-l", "researcher/10.0.0.1/self", "login.example.edu"]
    assert "BatchMode=yes" in args


def test_optional_alias():
    config = ClusterConfig(ssh_host_alias="school", remote_project_root="/project/agent")
    assert config.ssh_arguments()[-1] == "school"
    assert "-l" not in config.ssh_arguments()


def test_environment(monkeypatch):
    for key in ("LSF_SSH_HOST_ALIAS", "LSF_MEMORY_UNIT", "LSF_QUEUE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {"LSF_SSH_HOST": "login.example.edu", "LSF_SSH_PORT": "22",
                       "LSF_SSH_USERNAME": "researcher/10.0.0.1/self",
                       "LSF_REMOTE_PROJECT_ROOT": "/project/agent", "ENABLE_REAL_LSF": "true",
                       "LSF_SSH_INTERACTIVE": "true"}.items():
        monkeypatch.setenv(key, value)
    config = ClusterConfig.from_environment()
    assert config.enabled and config.interactive_auth
    assert config.ssh_username == "researcher/10.0.0.1/self"


@pytest.mark.parametrize("changes", [
    {"ssh_host": "host;id"}, {"ssh_host": "-oProxyCommand=bad"},
    {"ssh_username": "user$(id)"}, {"ssh_username": "user\nwhoami"},
    {"ssh_username": "user && id"}, {"ssh_username": "-bad"},
    {"ssh_port": 0}, {"ssh_port": 65536}, {"ssh_port": True},
    {"remote_project_root": "/project/../outside"}, {"remote_project_root": "/"},
    {"remote_project_root": "/project/agent;id"}, {"ssh_host_alias": "school"},
    {"ssh_username": None}, {"queue": "unapproved"},
])
def test_invalid_connection_rejected(changes):
    with pytest.raises(ValueError):
        direct(**changes)


def test_disabled_guard_never_calls_ssh(monkeypatch):
    monkeypatch.delenv("ENABLE_REAL_LSF", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("SSH must not run"))
    with pytest.raises(ValueError, match="disabled"):
        LSFBackend(None, None, direct()).check_connection()


@pytest.mark.parametrize("interactive", [False, True])
def test_ssh_transport_argv_and_password_not_captured(monkeypatch, interactive):
    monkeypatch.setenv("ENABLE_REAL_LSF", "true")
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"ready": true}', stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    backend = LSFBackend(None, None, direct(enabled=True, interactive_auth=interactive))
    assert backend.check_connection()["ready"]
    args, options = calls[0]
    assert args[args.index("-l") + 1] == "researcher/10.0.0.1/self"
    assert not options.get("shell", False)
    assert options["stderr"] == (None if interactive else subprocess.PIPE)
    assert "password" not in options["input"].lower()
