import pytest

import strands_solver.image_renderer.board_image_renderer as renderer


def test_themes_mapping_contains_light_and_dark() -> None:
    assert renderer.THEMES["light"] == renderer.LIGHT_THEME
    assert renderer.THEMES["dark"] == renderer.DARK_THEME


def test_renderer_uses_shared_board_validator() -> None:
    with pytest.raises(ValueError, match="less than minimum"):
        renderer.validate_board([])

    with pytest.raises(ValueError, match="less than minimum"):
        renderer.validate_board([""])

    with pytest.raises(ValueError, match="different from previous row"):
        renderer.validate_board(["abcd", "abcde", "abcd", "abcd"])


def test_build_coord_sets_filters_out_of_bounds_and_spangram_indexes() -> None:
    moves = [[(0, 0), (0, 1), (9, 9)], [(1, 0), (1, 1)]]

    word_coords, spangram_coords = renderer.build_coord_sets(moves, {1}, row_count=8, col_count=6)

    assert word_coords == {(0, 0), (0, 1)}
    assert spangram_coords == {(1, 0), (1, 1)}


@pytest.mark.skipif(not renderer.HAS_EXTRAS, reason="Pillow extras not installed")
def test_render_board_png_returns_png_and_centers() -> None:
    board = ["ABCDEF"] * 8

    image_bytes, centers = renderer.render_board_png(board, mode="light")

    assert image_bytes.startswith(b"\x89PNG")
    assert len(centers) == 48
    assert centers[(0, 0)][0] < centers[(0, 1)][0]


def test_render_board_png_raises_when_extras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(renderer, "HAS_EXTRAS", False)

    with pytest.raises(ModuleNotFoundError, match="Pillow is required"):
        renderer.render_board_png(["ABCD", "EFGH", "IJKL", "MNOP"], mode="light")
