#!/usr/bin/env python3

"""Print all valid sub-word compositions for a given word.

Usage:
    /home/tomas/strands_solver/.venv/bin/python tools/print_word_compositions.py \
        catsanddog /path/to/allowed_words.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_words(words_file: Path) -> set[str]:
    """Load non-empty normalized words from a text file."""
    with words_file.open() as handle:
        return {line.strip().lower() for line in handle if line.strip()}


def find_compositions(word: str, allowed_words: set[str]) -> list[tuple[str, ...]]:
    """Return all valid compositions of `word` using `allowed_words`."""
    memo: dict[int, list[tuple[str, ...]]] = {}

    def dfs(start_idx: int) -> list[tuple[str, ...]]:
        cached = memo.get(start_idx)
        if cached is not None:
            return cached

        if start_idx == len(word):
            memo[start_idx] = [()]
            return memo[start_idx]

        compositions: list[tuple[str, ...]] = []
        for end_idx in range(start_idx + 1, len(word) + 1):
            segment = word[start_idx:end_idx]
            if segment not in allowed_words:
                continue

            compositions.extend((segment, *tail) for tail in dfs(end_idx))

        memo[start_idx] = compositions
        return memo[start_idx]

    return dfs(0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Print all valid compositions of a word from an allowed-words file.",
    )
    parser.add_argument("word", help="Target word to segment")
    parser.add_argument("words_file", type=Path, help="Path to file containing allowed words, one per line")
    return parser.parse_args()


def main() -> None:
    """Run the word-composition CLI."""
    args = parse_args()
    word = args.word.strip().lower()
    allowed_words = load_words(args.words_file)
    compositions = find_compositions(word, allowed_words)

    if not compositions:
        print(f"No compositions found for: {word}")
        return

    for composition in compositions:
        print(" + ".join(composition))


if __name__ == "__main__":
    main()
