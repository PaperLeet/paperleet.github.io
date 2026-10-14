"""Startup and shutdown regressions, without binding ports or signalling real processes."""
import builtins
import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


class Process:
    def __init__(self, pid, code=None, timeout=False):
        self.pid = pid
        self.code = code
        self.timeout = timeout
        self.waits = []

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.timeout and len(self.waits) == 1:
            raise subprocess.TimeoutExpired("test service", timeout)
        return self.code or 0


@pytest.fixture
def dev(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/dev.py"
    spec = importlib.util.spec_from_file_location("paperleet_dev_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    handlers = {item: object() for item in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    original_handlers = handlers.copy()
    monkeypatch.setattr(module.signal, "getsignal", handlers.__getitem__)
    monkeypatch.setattr(module.signal, "signal", lambda item, handler: handlers.__setitem__(item, handler))
    monkeypatch.setattr(module, "preflight", lambda _: None)
    monkeypatch.setenv("PAPERLEET_AUTH_MODE", "test")
    # Startup may show the selected mode; it must never print credentials.
    monkeypatch.setenv("PAPERLEET_TEST_ACCOUNT_PASSWORD", "do-not-print-this-password")
    fake_export = ModuleType("paperleet.export")
    fake_export.export_catalog = lambda: None
    monkeypatch.setitem(sys.modules, "paperleet.export", fake_export)
    processes, groups = [], []

    def popen(*_, **kwargs):
        assert kwargs["start_new_session"] is True
        process = Process(120000 + len(processes))
        processes.append(process)
        return process

    def terminate_group(pid, item):
        assert all(handler == signal.SIG_IGN for handler in handlers.values())
        groups.append((pid, item))

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.os, "killpg", terminate_group)
    monkeypatch.setattr(module.time, "sleep", interrupt)
    return SimpleNamespace(module=module, export=fake_export, handlers=handlers,
                           original_handlers=original_handlers, processes=processes, groups=groups)


@pytest.mark.parametrize("phase", ["preflight", "import", "export", "first_spawn", "second_spawn", "running"])
def test_ctrl_c_is_clean_at_every_startup_phase(dev, monkeypatch, capsys, phase):
    def interrupt(*_, **__):
        raise KeyboardInterrupt

    if phase == "preflight":
        monkeypatch.setattr(dev.module, "preflight", interrupt)
    elif phase == "import":
        original_import = builtins.__import__

        def import_with_interrupt(name, *args, **kwargs):
            if name == "paperleet.export":
                raise KeyboardInterrupt
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", import_with_interrupt)
    elif phase == "export":
        dev.export.export_catalog = interrupt
    elif phase in ("first_spawn", "second_spawn"):
        original_popen = dev.module.subprocess.Popen

        def popen(*args, **kwargs):
            if phase == "first_spawn" or dev.processes:
                raise KeyboardInterrupt
            return original_popen(*args, **kwargs)

        monkeypatch.setattr(dev.module.subprocess, "Popen", popen)

    assert dev.module.main() == 0
    output = capsys.readouterr()
    assert output.out.startswith("Starting PaperLeet...")
    assert "Stopping PaperLeet" in output.out
    assert not output.err
    assert "do-not-print-this-password" not in output.out
    assert dev.groups == [(process.pid, signal.SIGTERM) for process in dev.processes]
    assert all(process.waits == [5] for process in dev.processes)
    assert dev.handlers == dev.original_handlers


@pytest.mark.parametrize("item", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_terminal_signals_are_handled_before_catalog_export(dev, item):
    dev.export.export_catalog = lambda: dev.handlers[item](item, None)
    assert dev.module.main() == 0
    assert not dev.processes
    assert dev.handlers == dev.original_handlers


def test_startup_status_is_flushed_before_application_import(dev, monkeypatch):
    prints = []
    original_print = builtins.print

    def record_print(*args, **kwargs):
        prints.append((args, kwargs))
        return original_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", record_print)
    original_import = builtins.__import__

    def check_import(name, *args, **kwargs):
        if name == "paperleet.export":
            assert prints[0][0][0].startswith("Starting PaperLeet")
            assert prints[-1][0] == ("Preparing the problem catalog...",)
            assert all(kwargs.get("flush") for _, kwargs in prints)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", check_import)
    assert dev.module.main() == 0
    assert any(args == ("Edition: local (no sign-in required)",) for args, _ in prints)


@pytest.mark.parametrize("service_index,code", [(0, 0), (0, 2), (1, -9)])
def test_unexpected_service_exit_fails_and_cleans_all_groups(dev, monkeypatch, capsys, service_index, code):
    popen = dev.module.subprocess.Popen

    def exiting_process(*args, **kwargs):
        process = popen(*args, **kwargs)
        if len(dev.processes) - 1 == service_index:
            process.code = code
        return process

    monkeypatch.setattr(dev.module.subprocess, "Popen", exiting_process)
    assert dev.module.main() == (code if code > 0 else 1)
    message = capsys.readouterr().err
    assert ("Judge" if service_index == 0 else "Web server") in message
    assert f"exit code {code}" in message
    # The exited group leader may still have children, so its group is also stopped.
    assert dev.groups == [(process.pid, signal.SIGTERM) for process in dev.processes]


def test_second_service_start_failure_cleans_first_service(dev, monkeypatch, capsys):
    popen = dev.module.subprocess.Popen

    def fail_second(*args, **kwargs):
        if dev.processes:
            raise FileNotFoundError("npm is unavailable")
        return popen(*args, **kwargs)

    monkeypatch.setattr(dev.module.subprocess, "Popen", fail_second)
    assert dev.module.main() == 1
    assert "npm is unavailable" in capsys.readouterr().err
    assert dev.groups == [(dev.processes[0].pid, signal.SIGTERM)]


def test_cleanup_escalates_stalled_owned_group_and_tolerates_missing_group(dev, monkeypatch):
    popen = dev.module.subprocess.Popen
    terminate_group = dev.module.os.killpg

    def stalled_process(*args, **kwargs):
        process = popen(*args, **kwargs)
        process.timeout = len(dev.processes) == 1
        return process

    def missing_group(pid, item):
        terminate_group(pid, item)
        if len(dev.processes) == 2 and pid == dev.processes[1].pid:
            raise ProcessLookupError

    monkeypatch.setattr(dev.module.subprocess, "Popen", stalled_process)
    monkeypatch.setattr(dev.module.os, "killpg", missing_group)
    assert dev.module.main() == 0
    assert dev.groups == [(process.pid, signal.SIGTERM) for process in dev.processes] + [(dev.processes[0].pid, signal.SIGKILL)]
    assert dev.processes[0].waits == [5, None]
    assert dev.handlers == dev.original_handlers


def test_occupied_port_is_reported_without_touching_existing_process(dev, monkeypatch, tmp_path, capsys):
    # Exercise the real preflight with fake sockets instead of contacting user services.
    path = tmp_path / "web/node_modules/.bin/vite"
    path.parent.mkdir(parents=True)
    path.touch()
    source = Path(__file__).resolve().parents[1] / "scripts/dev.py"
    spec = importlib.util.spec_from_file_location("paperleet_dev_preflight_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(dev.module, "preflight", lambda _: module.preflight(tmp_path))
    sockets = []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def connect_ex(self, address):
            sockets.append(address)
            return 0

    monkeypatch.setattr(module.socket, "socket", Socket)
    assert dev.module.main() == 1
    assert "Port 8765 is already in use" in capsys.readouterr().err
    assert sockets == [("127.0.0.1", 8765)]
    assert not dev.processes and not dev.groups
