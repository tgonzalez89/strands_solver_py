"""CLI for rendering Strands board-state screenshots."""

import argparse
from pathlib import Path

from strands_solver.image_renderer.board_image_renderer import RenderConfig, build_coord_sets, render_board_png
from strands_solver.util.util import load_board, load_moves


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse board-image CLI arguments.

    Args:
        argv: Optional explicit argv list. When omitted, parses from process args.

    Returns:
        Parsed argument namespace.

    """
    parser = argparse.ArgumentParser(description="Render a Strands board image from board/moves fixtures.")
    parser.add_argument("--board", type=Path, required=True, help="Path to board file.")
    parser.add_argument("--valid-moves", type=Path, help="Optional path to valid moves file.")
    parser.add_argument(
        "--spangram-index",
        type=int,
        action="append",
        default=[],
        help="0-based move index to render as spangram. Can be provided multiple times.",
    )
    parser.add_argument("--mode", choices=("light", "dark"), default="dark", help="Color mode.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--width", type=int, default=1080, help="Image width in pixels.")
    parser.add_argument("--height", type=int, default=2246, help="Image height in pixels.")
    parser.add_argument("--board-width-ratio", type=float, default=0.74, help="Board width / image width.")
    parser.add_argument("--board-height-ratio", type=float, default=0.53, help="Board height / image height.")
    parser.add_argument(
        "--board-center-y-ratio",
        type=float,
        default=0.54,
        help="Board center Y location / image height.",
    )
    parser.add_argument("--cell-radius-ratio", type=float, default=0.42, help="Cell radius / min(cell dimensions).")
    parser.add_argument("--font-size-ratio", type=float, default=0.48, help="Font size / min(cell dimensions).")
    parser.add_argument("--font-path", type=Path, help="Optional TTF font path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run board-image rendering CLI.

    Args:
        argv: Optional explicit argv list. When omitted, parses from process args.

    Returns:
        Process exit code.

    """
    args = parse_args(argv)

    board = load_board(args.board)
    moves = load_moves(args.valid_moves) if args.valid_moves is not None else []
    spangram_indexes = {idx for idx in args.spangram_index if idx >= 0}

    word_coords, spangram_coords = build_coord_sets(
        moves,
        spangram_indexes,
        row_count=len(board),
        col_count=len(board[0]),
    )
    png_bytes, _ = render_board_png(
        board,
        mode=args.mode,
        word_coords=word_coords,
        spangram_coords=spangram_coords,
        config=RenderConfig(
            width=args.width,
            height=args.height,
            board_width_ratio=args.board_width_ratio,
            board_height_ratio=args.board_height_ratio,
            board_center_y_ratio=args.board_center_y_ratio,
            cell_radius_ratio=args.cell_radius_ratio,
            font_size_ratio=args.font_size_ratio,
            font_path=args.font_path,
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(png_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
