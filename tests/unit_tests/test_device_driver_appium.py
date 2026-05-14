import pytest

from strands_solver.device.device_driver_appium import DeviceDriverAppium


class MockAppiumSession:
    def __init__(self, screenshot: bytes = b"png") -> None:
        self._screenshot = screenshot
        self.swipes: list[tuple[int, int, int, int, int]] = []

    def get_screenshot_as_png(self) -> bytes:
        return self._screenshot

    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int) -> None:
        self.swipes.append((start_x, start_y, end_x, end_y, duration))


def test_appium_driver_capture_screen_returns_png_bytes() -> None:
    driver = DeviceDriverAppium(session=MockAppiumSession(screenshot=b"image-bytes"))

    assert driver.capture_screen() == b"image-bytes"


def test_appium_driver_execute_path_swipes_all_segments() -> None:
    session = MockAppiumSession()
    driver = DeviceDriverAppium(session=session, swipe_duration_ms=250)

    driver.execute_path([(10, 20), (30, 40), (50, 60)])

    assert session.swipes == [
        (10, 20, 30, 40, 250),
        (30, 40, 50, 60, 250),
    ]


def test_appium_driver_execute_path_single_point_uses_degenerate_swipe() -> None:
    session = MockAppiumSession()
    driver = DeviceDriverAppium(session=session)

    driver.execute_path([(10, 20)])

    assert session.swipes == [(10, 20, 10, 20, 10)]


def test_appium_driver_execute_path_raises_for_empty_path() -> None:
    driver = DeviceDriverAppium(session=MockAppiumSession())

    with pytest.raises(ValueError, match="pixel_path must contain at least one coordinate"):
        driver.execute_path([])


def test_appium_driver_raises_when_session_not_configured() -> None:
    driver = DeviceDriverAppium()

    with pytest.raises(NotImplementedError, match="session is not configured"):
        driver.capture_screen()
