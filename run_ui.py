"""Launch the ScholarGraph Control Deck web UI.

If another instance is already bound to WEB_PORT, stop it first.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time


def _listening_pids(port: int) -> set[int]:
    """Return PIDs that are LISTENING on the given TCP port."""
    pids: set[int] = set()
    me = os.getpid()
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "tcp"],
                text=True,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return pids
        needle = f":{port}"
        for line in out.splitlines():
            if "LISTENING" not in line.upper():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            local = parts[1]
            # Match :8765 at end of address (IPv4 or [::]:8765)
            if not (local.endswith(needle) or local.endswith(f"]{needle}")):
                continue
            try:
                pid = int(parts[-1])
            except ValueError:
                continue
            if pid > 0 and pid != me:
                pids.add(pid)
        return pids

    # Unix: prefer lsof, fall back to fuser
    for cmd in (
        ["lsof", "-ti", f"TCP:{port}", "-sTCP:LISTEN"],
        ["fuser", f"{port}/tcp"],
    ):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            continue
        for token in out.replace("\n", " ").split():
            try:
                pid = int(token)
            except ValueError:
                continue
            if pid > 0 and pid != me:
                pids.add(pid)
        if pids:
            break
    return pids


def free_port(port: int, host: str = "127.0.0.1") -> None:
    """Kill any process listening on host:port so this launcher can bind."""
    pids = _listening_pids(port)
    if not pids:
        return
    print(f"Port {port} busy — stopping previous Control Deck instance(s): {sorted(pids)}")
    for pid in sorted(pids):
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    check=False,
                    capture_output=True,
                    text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                os.kill(pid, 15)  # SIGTERM
        except OSError as exc:
            print(f"  could not stop PID {pid}: {exc}")

    # Wait briefly for the OS to release the socket
    deadline = time.time() + 5.0
    while time.time() < deadline:
        remaining = _listening_pids(port)
        if not remaining:
            print(f"Port {port} is free.")
            return
        time.sleep(0.25)
    leftover = _listening_pids(port)
    if leftover:
        print(f"Warning: still listening on {port}: {sorted(leftover)}")


def main() -> None:
    from core.config import config
    import os

    os.makedirs("memory/vector_db", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Sandbox readiness (fast check, no build — never blocks UI startup).
    # Ephemeral model: experiments spawn `docker run --rm` per call, so no
    # persistent container is expected; only Docker Desktop + the image matter.
    try:
        from core.container_sandbox import sandbox_status as _sandbox_status

        _sb = _sandbox_status()
        if _sb.get("ready"):
            print(f"Sandbox: docker ready ({_sb.get('image')}) — ephemeral containers per experiment.")
        else:
            print(f"Sandbox: NOT ready ({_sb.get('reason', 'unknown')}). {_sb.get('hint', '')}")
            print("  Experiments will use the AST fallback until Docker is ready.")
            print("  Check/fix with: python -m core.container_sandbox [--build]")
    except Exception as exc:
        print(f"Sandbox check skipped: {exc}")

    free_port(config.web_port, config.web_host)
    from web.app import main as run_app

    run_app()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python run_ui.py [--host HOST] [--port PORT]")
        print("Launches the ScholarGraph Control Deck web UI (no other flags).")
        print("Host/port default from WEB_HOST/WEB_PORT in .env (127.0.0.1:8765).")
        sys.exit(0)
    overrides = {}
    for flag, key in (("--host", "WEB_HOST"), ("--port", "WEB_PORT")):
        if flag in sys.argv:
            try:
                overrides[key] = sys.argv[sys.argv.index(flag) + 1]
            except IndexError:
                print(f"Missing value for {flag}", file=sys.stderr)
                sys.exit(2)
    if overrides:
        os.environ.update(overrides)
    main()
