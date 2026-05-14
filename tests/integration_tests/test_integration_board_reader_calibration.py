from typing import TYPE_CHECKING

import pytest

import strands_solver.board_reader.board_reader_tesseract_open_cv as board_reader_module
from strands_solver.device.device_driver import DeviceDriver
from strands_solver.image_renderer.board_image_renderer import HAS_EXTRAS as HAS_RENDERER_EXTRAS
from strands_solver.image_renderer.board_image_renderer import render_board_png

if TYPE_CHECKING:
    from strands_solver.util.util import PixelCoord


class _ScriptedDriver(DeviceDriver):
    def __init__(self, screenshots: list[bytes]) -> None:
        self._screenshots = screenshots
        self.taps: list[PixelCoord] = []

    def capture_screen(self) -> bytes:
        if not self._screenshots:
            msg = "No more scripted screenshots available"
            raise AssertionError(msg)
        return self._screenshots.pop(0)

    def tap(self, coord: PixelCoord) -> None:
        self.taps.append(coord)

    def execute_path(self, pixel_path: list[PixelCoord]) -> None:
        _ = pixel_path


@pytest.mark.skipif(not board_reader_module.HAS_EXTRAS, reason="OpenCV + tesserocr extras not installed")
@pytest.mark.skipif(not HAS_RENDERER_EXTRAS, reason="Pillow extras not installed")
def test_integration_calibrate_updates_reader_cell_centers_from_detected_selection_blobs() -> None:
    board = [
        "ABCDEF",
        "GHIJKL",
        "MNOPQR",
        "STUVWX",
        "YZABCD",
        "EFGHIJ",
        "KLMNOP",
        "QRSTUV",
    ]

    clear_screen, centers = render_board_png(board, mode="light")
    top_left_selected_screen, _ = render_board_png(board, mode="light", selection_coords={(0, 0)})
    bottom_right_selected_screen, _ = render_board_png(board, mode="light", selection_coords={(7, 5)})

    expected_top_left = centers[(0, 0)]
    expected_bottom_right = centers[(7, 5)]
    expected_circle_diameter = 2 * int(min(153.0, 148.5) * 0.42)

    # Expected capture sequence by calibrate():
    # 1) ensure clear before top-left prompt
    # 2) poll top-left tap (one selected blob)
    # 3) ensure clear before bottom-right prompt
    # 4) poll bottom-right tap (one selected blob)
    driver = _ScriptedDriver(
        [
            clear_screen,
            top_left_selected_screen,
            clear_screen,
            bottom_right_selected_screen,
        ],
    )
    reader = board_reader_module.BoardReaderTesseractOpenCV(rows=8, cols=6)

    reader.calibrate(driver, timeout_s=2.0, poll_interval_s=0.0)

    calibrated_top_left = reader._cell_centers[0][0]
    calibrated_bottom_right = reader._cell_centers[7][5]

    assert abs(calibrated_top_left[0] - expected_top_left[0]) <= 3
    assert abs(calibrated_top_left[1] - expected_top_left[1]) <= 3
    assert abs(calibrated_bottom_right[0] - expected_bottom_right[0]) <= 3
    assert abs(calibrated_bottom_right[1] - expected_bottom_right[1]) <= 3
    assert abs(reader.get_estimated_circle_diameter() - expected_circle_diameter) <= 3
    assert driver.taps == [expected_top_left, expected_bottom_right]
    assert driver._screenshots == []
