# Lightweight Remote LSF acceptance

## Authenticated session reuse: blocked on this Windows client

The intended design is manual password authentication once, followed by Agent
reuse of that authenticated connection, with no application-stored credentials.
It is **not implemented or verified on the current Windows client**.

Actual local capability checks found:

- Executable: system OpenSSH, version `OpenSSH_for_Windows_8.6p1`, LibreSSL 3.4.3.
- `ssh -G` accepts and displays ControlMaster, ControlPath and ControlPersist.
  Configuration parsing alone does not establish working multiplexing.
- `ssh -S <local-probe-path> -O check <configured-host>` exits nonzero with
  `getsockname failed: Not a socket`. The same result occurs outside the sandbox.
- The [official Win32-OpenSSH project scope](https://github.com/PowerShell/Win32-OpenSSH/wiki/Project-Scope)
  lists Client ControlMaster as unsupported on Windows.

An ordinary interactive SSH terminal cannot be reused by a separate Backend
subprocess. Consequently there are no open/check/close master scripts or active
session-reuse environment flags in this project: publishing them for this client
would imply functionality that has not been established.

The intended lifecycle, once a supported transport is available, is:

1. Establish an authenticated connection manually in the user's terminal.
2. Verify that its control endpoint is reusable without authentication prompts.
3. Run the Agent/LSFBackend smoke against that same connection.
4. Explicitly close the connection when finished.

For this client, replacing ControlMaster with a persistent SSH process shared
across independently launched programs would require local IPC ownership,
request/output framing, timeouts and disconnect handling. That exceeds the agreed
minimal fallback scope, so implementation stops here. No password automation,
key setup, authentication configuration change or real LSF submission was made.
This historical investigation changed documentation only. See the main README
for current test results; no session regression coverage was added.

This acceptance probe uses the existing LSFBackend and SQLite ownership records.
It runs no model, dataset, PyTorch, DeepSeek or external-network task.

## Local configuration

Use process variables or the ignored repository `.env`:

```env
ENABLE_REAL_LSF=true
LSF_SSH_HOST=<hostname>
LSF_SSH_PORT=22
LSF_SSH_USERNAME=<opaque-username>
LSF_REMOTE_PROJECT_ROOT=<absolute-project-root>
```

Only allowlisted connection settings are loaded by `run_lsf_smoke.py`; password,
private-key and token variables are ignored. Username slashes are passed unchanged
as one `ssh -l` argument. An optional `LSF_SSH_HOST_ALIAS` can replace direct host
and username settings. Real values must never be copied into public documentation.

StrictHostKeyChecking=yes applies even with `--interactive`. The backend never
accepts new host keys or changes known_hosts. `--interactive` lets native OpenSSH
ask for authentication in the user's terminal; Python never handles the password.
The API remains noninteractive. An earlier successful manual SSH login does not
provide credentials to a later independent backend SSH process.

## Remote prerequisites

The configured root must already exist. Place the reviewed local file
`toy_project/lsf_smoke.py` at that exact relative path under the remote root.
Python 3 must be available in noninteractive SSH and at the same path on compute
nodes. The probe uses only the standard library and needs no venv or dependencies.
The agent does not automatically copy the repository, install software or edit
server configuration. All generated config, logs and the text artifact remain
under the configured root; symlink escapes are rejected.

## Resources

Queues are restricted to `734090d!`, `944090d!`, `9654p6000ib!`. No GPU/hardware,
memory or queue capability is inferred from these names. The probe requests one
CPU core, no GPU, and a one-minute LSF runtime limit. It intentionally omits both
`-M` and `rusage[mem=...]`, using the site's default memory rather than guessing
an undocumented unit. Those defaults are not a guaranteed application memory cap.
If the scheduler requires an explicit memory request, stop and ask the operator
for confirmed units/requirements; do not silently retry with guessed values.
Ordinary training jobs still require an explicit confirmed LSF_MEMORY_UNIT.

## Run

```powershell
python scripts/run_lsf_smoke.py --check-only
python scripts/run_lsf_smoke.py --interactive
python scripts/run_lsf_smoke.py --interactive --attempt-id '<existing-attempt-id>'
```

The script submits through ExperimentService → LSFBackend, reconstructs the backend
before polling to verify persistent ownership, polls at the configured interval,
reads labelled stdout/stderr and checks `smoke_result.txt`. It waits five minutes
by default; unfinished jobs retain their Attempt/job ID for explicit continuation.
It never creates a replacement when resuming an existing Attempt. Evidence is
saved under ignored `data/lsf_smoke_validation.json`; logs/artifact stay in the
local attempt directory. Public docs contain no host, username or remote root.

Only PASS after actual remote completion permits this claim:
`Remote LSFBackend validated against a real LSF cluster using a lightweight job.`
It would not establish real ML training or DeepSeek end-to-end validation.

## Actual acceptance status

The latest LSFBackend readiness check passed host-key verification, but OpenSSH
rejected noninteractive authentication (password/keyboard-interactive/publickey).
The user reports successful manual login; that is distinct from backend success.
No real job was submitted: job ID parsing, bjobs, remote stdout/stderr and artifact
retrieval remain unverified. This is an authentication environment block, not a
reported scheduler failure. No password was requested by or stored in Python.
