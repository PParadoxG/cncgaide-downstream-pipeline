"""Find, identify and stop the local viewer servers.

The viewer deliberately refuses to share a port: ``SO_REUSEADDR`` on Windows
would let a second server bind a port another process still holds, and the
page would then quietly show somebody else's run.  Failing is right; failing
with nothing but "is a viewer already running?" is not, because the person in
front of the terminal has no way to find the process that is in the way — it
may have been started by another shell, another session, or an agent.

This module supplies the missing half: who holds the port, what data they are
serving, and how to stop them.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

APP_ID = "gatherseed-viewer"
LOCK_DIR = ".viewer"
SHUTDOWN_HEADER = "X-Viewer-Shutdown"
# Environment label, so a server started by a script or an agent says so.
LABEL_ENV = "GATHERSEED_VIEWER_LABEL"


def lock_root(project_root: Path) -> Path:
    """Where instance records live.

    Not under ``runs/<tag>/``: a refresh wipes and rebuilds that directory,
    which would delete the record of the very server that started the refresh.
    """
    return Path(project_root) / LOCK_DIR


def lock_path(project_root: Path, port: int) -> Path:
    return lock_root(project_root) / f"{int(port)}.json"


def write_lock(project_root: Path, *, port: int, output_tag: str, run_root: Path) -> Path:
    root = lock_root(project_root)
    root.mkdir(parents=True, exist_ok=True)
    path = lock_path(project_root, port)
    path.write_text(json.dumps(identity(port=port, output_tag=output_tag, run_root=run_root), indent=2), encoding="utf-8")
    return path


def clear_lock(project_root: Path, port: int) -> None:
    try:
        lock_path(project_root, port).unlink()
    except OSError:
        pass


def identity(*, port: int, output_tag: str, run_root: Path) -> dict:
    return {
        "app": APP_ID,
        "pid": os.getpid(),
        "port": int(port),
        "outputTag": output_tag,
        "runRoot": str(run_root),
        "startedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "startedAtEpoch": time.time(),
        "label": os.environ.get(LABEL_ENV, ""),
    }


def probe(port: int, timeout: float = 1.0) -> dict | None:
    """Ask whatever holds the port to identify itself.

    Returns the viewer's own status payload, or ``None`` if nothing answered
    or the answer did not come from a viewer.
    """
    import http.client

    try:
        connection = http.client.HTTPConnection("127.0.0.1", int(port), timeout=timeout)
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
    except Exception:
        return None
    return payload if isinstance(payload, dict) and payload.get("app") == APP_ID else None


def port_owner_pid(port: int) -> int | None:
    """PID holding the port, for the case where it is not one of ours."""
    try:
        if sys.platform == "win32":
            output = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=10
            ).stdout
            pattern = re.compile(rf"127\.0\.0\.1:{int(port)}\s+\S+\s+LISTENING\s+(\d+)")
            match = pattern.search(output)
            return int(match.group(1)) if match else None
        output = subprocess.run(
            ["lsof", "-ti", f"tcp:{int(port)}", "-sTCP:LISTEN"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return int(output.splitlines()[0]) if output else None
    except Exception:
        return None


def stop(port: int, timeout: float = 5.0) -> tuple[bool, str]:
    """Ask the viewer on *port* to shut itself down.

    Only ever a polite request over the server's own control endpoint: no
    signal is sent, so a recycled PID or an unrelated program on that port
    cannot be killed by accident.
    """
    import http.client

    if probe(port) is None:
        return False, "no GatherSeed viewer is answering on that port"
    try:
        connection = http.client.HTTPConnection("127.0.0.1", int(port), timeout=timeout)
        connection.request("POST", "/api/shutdown", headers={SHUTDOWN_HEADER: "1"})
        connection.getresponse().read()
        connection.close()
    except Exception as error:
        return False, f"shutdown request failed: {type(error).__name__}: {error}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if probe(port, timeout=0.5) is None:
            return True, "stopped"
        time.sleep(0.2)
    return False, "the viewer did not stop within the timeout"


def discover(project_root: Path, extra_ports: list[int] | None = None) -> list[dict]:
    """Every viewer we can find, with its lock file reconciled against reality."""
    ports: list[int] = []
    for path in sorted(lock_root(project_root).glob("*.json")):
        try:
            ports.append(int(path.stem))
        except ValueError:
            continue
    for port in extra_ports or []:
        if int(port) not in ports:
            ports.append(int(port))

    found = []
    for port in ports:
        status = probe(port)
        if status is None:
            # Nothing is answering: the process died without cleaning up.
            clear_lock(project_root, port)
            continue
        found.append(status)
    return found


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        return int(probe_socket.getsockname()[1])


def describe_conflict(port: int) -> list[str]:
    """The actionable half of a failed bind."""
    lines = [f"[viewer] 127.0.0.1:{port} is already in use"]
    status = probe(port)
    if status is not None:
        age = ""
        started = status.get("startedAtEpoch")
        if isinstance(started, (int, float)):
            minutes = max(0, int((time.time() - started) // 60))
            age = f" ({minutes // 60}h{minutes % 60:02d}m ago)" if minutes >= 60 else f" ({minutes}m ago)"
        label = status.get("label") or ""
        lines += [
            f"           holder  GatherSeed viewer - pid {status.get('pid')} - tag={status.get('outputTag')}"
            + (f" - label={label}" if label else ""),
            f"                   data {status.get('runRoot')}",
            f"                   started {status.get('startedAt')}{age}",
            f"           stop it     run_pipeline.py view --stop --port {port}",
            f"           take it     run_pipeline.py view --replace --port {port}",
            f"           or move     run_pipeline.py view --port {port + 1}",
        ]
        return lines
    pid = port_owner_pid(port)
    lines.append(
        f"           holder  not a GatherSeed viewer - pid {pid}" if pid
        else "           holder  could not be identified"
    )
    lines.append(f"           try     run_pipeline.py view --port {port + 1}   (or --port auto)")
    return lines
