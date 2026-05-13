import subprocess

import pytest

from strands_solver.device.device_driver_adb import DeviceDriverADB


def test_adb_driver_capture_screen_returns_png_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        _ = args, kwargs
        return subprocess.CompletedProcess(args=["adb"], returncode=0, stdout=b"png-bytes", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    driver = DeviceDriverADB()

    assert driver.capture_screen() == b"png-bytes"


def test_adb_driver_capture_screen_normalizes_line_endings(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        _ = args, kwargs
        return subprocess.CompletedProcess(args=["adb"], returncode=0, stdout=b"A\r\nB\r\n", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    driver = DeviceDriverADB()

    assert driver.capture_screen() == b"A\nB\n"


def test_adb_driver_execute_path_swipes_all_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        _ = kwargs
        commands.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    driver = DeviceDriverADB(device_serial="SER123", swipe_duration_ms=250)

    driver.execute_path([(10, 20), (30, 40), (50, 60)])

    assert commands == [
        ["adb", "-s", "SER123", "shell", "input", "swipe", "10", "20", "30", "40", "250"],
        ["adb", "-s", "SER123", "shell", "input", "swipe", "30", "40", "50", "60", "250"],
    ]


def test_adb_driver_execute_path_includes_server_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        _ = kwargs
        commands.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    driver = DeviceDriverADB(adb_server_host="host.docker.internal", adb_server_port=5037)

    driver.execute_path([(10, 20), (30, 40)])

    assert commands == [
        [
            "adb",
            "-H",
            "host.docker.internal",
            "-P",
            "5037",
            "shell",
            "input",
            "swipe",
            "10",
            "20",
            "30",
            "40",
            "120",
        ],
    ]


def test_adb_driver_execute_path_single_point_uses_degenerate_swipe(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        _ = kwargs
        commands.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    driver = DeviceDriverADB(swipe_duration_ms=180)

    driver.execute_path([(10, 20)])

    assert commands == [["adb", "shell", "input", "swipe", "10", "20", "10", "20", "180"]]


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
        raise subprocess.TimeoutExpired(cmd=["adb"], timeout=15)

    monkeypatch.setattr(subprocess, "run", fake_run)
    driver = DeviceDriverADB(command_timeout_s=15)

    with pytest.raises(NotImplementedError, match="adb command timed out"):
        driver.capture_screen()
