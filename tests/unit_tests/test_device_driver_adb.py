import subprocess

import pytest

from strands_solver.device.device_driver_adb import DeviceDriverADB


def _make_fake_run(commands: list[list[str]]) -> object:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        _ = kwargs
        commands.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=b"", stderr=b"")

    return fake_run


def test_adb_driver_capture_screen_returns_png_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        _ = args, kwargs
        return subprocess.CompletedProcess(args=["adb"], returncode=0, stdout=b"png-bytes", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    driver = DeviceDriverADB()

    assert driver.capture_screen() == b"png-bytes"


def test_adb_driver_execute_path_taps_all_cells_plus_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(subprocess, "run", _make_fake_run(commands))
    monkeypatch.setattr("strands_solver.device.device_driver_adb.time.sleep", lambda _: None)
    driver = DeviceDriverADB(device_serial="SER123", tap_delay_ms=250)

    driver.execute_path([(10, 20), (30, 40), (50, 60)])

    # 3 cells + 1 confirmation tap on the last cell = 4 commands total.
    assert commands == [
        ["adb", "-s", "SER123", "shell", "input", "tap", "10", "20"],
        ["adb", "-s", "SER123", "shell", "input", "tap", "30", "40"],
        ["adb", "-s", "SER123", "shell", "input", "tap", "50", "60"],
        ["adb", "-s", "SER123", "shell", "input", "tap", "50", "60"],  # confirmation
    ]


def test_adb_driver_execute_path_single_point_taps_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-cell path should still emit two taps (select + confirm)."""
    commands: list[list[str]] = []

    monkeypatch.setattr(subprocess, "run", _make_fake_run(commands))
    monkeypatch.setattr("strands_solver.device.device_driver_adb.time.sleep", lambda _: None)
    driver = DeviceDriverADB(tap_delay_ms=180)

    driver.execute_path([(10, 20)])

    assert commands == [
        ["adb", "shell", "input", "tap", "10", "20"],  # select
        ["adb", "shell", "input", "tap", "10", "20"],  # confirm
    ]


def test_adb_driver_execute_path_includes_server_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(subprocess, "run", _make_fake_run(commands))
    monkeypatch.setattr("strands_solver.device.device_driver_adb.time.sleep", lambda _: None)
    driver = DeviceDriverADB(adb_server_host="host.docker.internal", adb_server_port=5037)

    driver.execute_path([(10, 20), (30, 40)])

    # Both tap commands must include -H / -P flags.
    for cmd in commands:
        assert "-H" in cmd
        assert "host.docker.internal" in cmd
        assert "-P" in cmd
        assert "5037" in cmd


def test_adb_driver_execute_path_raises_for_empty_path() -> None:
    driver = DeviceDriverADB()

    with pytest.raises(ValueError, match="pixel_path must contain at least one coordinate"):
        driver.execute_path([])


def test_adb_driver_raises_when_adb_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        _ = args, kwargs
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)
    driver = DeviceDriverADB()

    with pytest.raises(NotImplementedError, match="adb executable not found"):
        driver.capture_screen()


def test_adb_driver_raises_when_command_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        _ = args, kwargs
        return subprocess.CompletedProcess(args=["adb"], returncode=1, stdout=b"", stderr=b"device not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    driver = DeviceDriverADB()

    with pytest.raises(NotImplementedError, match="adb command failed: device not found"):
        driver.capture_screen()


def test_adb_driver_raises_when_command_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        _ = args, kwargs
        raise subprocess.TimeoutExpired(cmd=["adb"], timeout=10)

    monkeypatch.setattr(subprocess, "run", fake_run)
    driver = DeviceDriverADB(command_timeout_s=10)

    with pytest.raises(NotImplementedError, match="adb command timed out"):
        driver.capture_screen()
