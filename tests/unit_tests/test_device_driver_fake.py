from typing import TYPE_CHECKING

import strands_solver.device.device_driver_fake as fake_driver_module
from strands_solver.device.device_driver_fake import DeviceDriverFake
from strands_solver.image_renderer.board_image_renderer import DARK_THEME, LIGHT_THEME, RenderConfig, render_board_png

if TYPE_CHECKING:
    import pytest

    from strands_solver.util.util import BoardCoord, PixelCoord


def _board() -> list[str]:
    return [
        "abcdef",
        "ghijkl",
        "mnopqr",
        "stuvwx",
        "yzabcd",
        "efghij",
        "klmnop",
        "qrstuv",
    ]


def test_fake_driver_capture_screen_returns_png_bytes() -> None:
    driver = DeviceDriverFake(
        board=_board(),
        valid_moves=[[(0, 0), (0, 1), (0, 2), (0, 3)]],
    )

    screenshot = driver.capture_screen()

    assert screenshot.startswith(b"\x89PNG")


def test_fake_driver_execute_valid_path_changes_next_screenshot() -> None:
    board = _board()
    valid_move = [(0, 0), (0, 1), (0, 2), (0, 3)]
    driver = DeviceDriverFake(board=board, valid_moves=[valid_move])

    before = driver.capture_screen()
    _, centers = render_board_png(board, config=RenderConfig(theme=LIGHT_THEME))
    pixel_path = [centers[coord] for coord in valid_move]

    driver.execute_path(pixel_path)
    after = driver.capture_screen()

    assert before != after


def test_fake_driver_execute_invalid_path_keeps_screenshot_same() -> None:
    board = _board()
    valid_move = [(0, 0), (0, 1), (0, 2), (0, 3)]
    invalid_move = [(1, 0), (1, 1), (1, 2), (1, 3)]
    driver = DeviceDriverFake(
        board=board,
        valid_moves=[valid_move],
        render_config=RenderConfig(theme=DARK_THEME),
    )

    before = driver.capture_screen()
    _, centers = render_board_png(board, config=RenderConfig(theme=DARK_THEME))
    pixel_path = [centers[coord] for coord in invalid_move]

    driver.execute_path(pixel_path)
    after = driver.capture_screen()

    assert before == after


def test_fake_driver_capture_screen_reuses_cached_render(monkeypatch: pytest.MonkeyPatch) -> None:
    board = _board()
    driver = DeviceDriverFake(board=board, valid_moves=[[(0, 0), (0, 1), (0, 2), (0, 3)]])

    render_calls = 0
    original_render = fake_driver_module.render_board_png

    def wrapped_render(
        board_rows: list[str],
        *,
        word_coords: set[BoardCoord] | None = None,
        spangram_coords: set[BoardCoord] | None = None,
        config: RenderConfig | None = None,
    ) -> tuple[bytes, dict[BoardCoord, PixelCoord]]:
        nonlocal render_calls
        render_calls += 1
        return original_render(
            board_rows,
            word_coords=word_coords,
            spangram_coords=spangram_coords,
            config=config,
        )

    monkeypatch.setattr(fake_driver_module, "render_board_png", wrapped_render)

    first = driver.capture_screen()
    second = driver.capture_screen()

    assert first == second
    assert render_calls == 1
