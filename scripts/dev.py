"""One command starts the local judge and Vite, then cleans up both on Ctrl-C."""
import signal
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def preflight(repository):
    if not (repository / "web/node_modules/.bin/vite").exists():
        raise RuntimeError("Run `cd web && npm ci` first.")
    for port in (8765, 5173):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Port {port} is already in use. Stop the existing dev process first.")


def main():
    processes = []
    previous_handlers = {}
    stopping = False

    def stop(*_):
        if not stopping:
            raise KeyboardInterrupt

    shutdown_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        shutdown_signals.append(signal.SIGHUP)

    try:
        for shutdown_signal in shutdown_signals:
            previous_handlers[shutdown_signal] = signal.getsignal(shutdown_signal)
            signal.signal(shutdown_signal, stop)
        print("Starting PaperLeet... (Ctrl-C to stop)", flush=True)
        repository = Path(__file__).resolve().parents[1]
        preflight(repository)

        # Application imports belong inside the interrupt-controlled startup.
        from paperleet.config import validate_local_settings
        validate_local_settings()
        print("Edition: local (no sign-in required)", flush=True)
        print("Preparing the problem catalog...", flush=True)
        from paperleet.export import export_catalog
        export_catalog()

        print("Starting judge on 127.0.0.1:8765 and web on http://127.0.0.1:5173...", flush=True)
        processes.append(("Judge", subprocess.Popen([sys.executable, "-m", "paperleet.cli", "serve"], cwd=repository, start_new_session=True)))
        processes.append(("Web server", subprocess.Popen(["npm", "run", "dev", "--", "--port", "5173", "--strictPort"], cwd=repository / "web", start_new_session=True)))
        while True:
            for name, process in processes:
                code = process.poll()
                if code is not None:
                    print(f"{name} exited unexpectedly (exit code {code}); stopping PaperLeet.", file=sys.stderr, flush=True)
                    return code if code > 0 else 1
            time.sleep(.25)
    except KeyboardInterrupt:
        print("\nStopping PaperLeet...", flush=True)
        return 0
    except Exception as error:
        print(f"PaperLeet startup failed: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        # A second terminal signal must not interrupt child cleanup.
        stopping = True
        for shutdown_signal in previous_handlers:
            signal.signal(shutdown_signal, signal.SIG_IGN)
        try:
            for _, process in processes:
                try:
                    # npm may have exited while its Vite child still owns the port.
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for _, process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        finally:
            for shutdown_signal, previous_handler in previous_handlers.items():
                signal.signal(shutdown_signal, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
