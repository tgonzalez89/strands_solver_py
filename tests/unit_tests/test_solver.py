from strands_solver.solver.solver import Trie, get_neighbor_coords, leaves_small_island


def test_get_neighbor_coords_center_has_8_neighbors() -> None:
    board = [
        "abcd",
        "efgh",
        "ijkl",
        "mnop",
    ]

    neighbors = get_neighbor_coords(board, 1, 1)

    assert len(neighbors) == 8
    assert (0, 0) in neighbors
    assert (2, 2) in neighbors


def test_get_neighbor_coords_corner_has_3_neighbors() -> None:
    board = [
        "abcd",
        "efgh",
        "ijkl",
        "mnop",
    ]

    neighbors = get_neighbor_coords(board, 0, 0)

    assert sorted(neighbors) == [(0, 1), (1, 0), (1, 1)]


def test_trie_build_from_words_finds_horizontal_path() -> None:
    board = [
        "test",
        "abcd",
        "rate",
        "wxyz",
    ]
    trie = Trie.build_from_words(["test"])

    paths = trie.find_all_word_paths(board)

    assert [(0, 0), (0, 1), (0, 2), (0, 3)] in paths


def test_trie_finds_diagonal_word_path() -> None:
    board = [
        "aqqq",
        "qmqx",
        "qqaq",
        "qqqx",
    ]
    trie = Trie.build_from_words(["amax"])

    paths = trie.find_all_word_paths(board)

    assert tuple(paths[0]) in {
        ((0, 0), (1, 1), (2, 2), (3, 3)),
        ((0, 0), (1, 1), (2, 2), (1, 3)),
    }


def test_trie_discards_self_crossing_path() -> None:
    board = [
        "abcd",
        "efgh",
    ]
    trie = Trie.build_from_words(["afeb"])

    paths = trie.find_all_word_paths(board)

    assert [(0, 0), (1, 1), (1, 0), (0, 1)] not in paths


def test_trie_returns_empty_when_no_words_match() -> None:
    board = [
        "test",
        "abcd",
        "rate",
        "wxyz",
    ]
    trie = Trie.build_from_words(["nope"])

    assert trie.find_all_word_paths(board) == []


def test_trie_does_not_reuse_same_coordinate_in_path() -> None:
    board = [
        "aa",
        "aa",
    ]
    trie = Trie.build_from_words(["aaaaa"])

    assert trie.find_all_word_paths(board) == []


def test_trie_returns_both_prefix_and_longer_word_paths() -> None:
    board = [
        "tests",
        "abcde",
        "fghij",
        "klmno",
        "pqrst",
    ]
    trie = Trie.build_from_words(["test", "tests"])

    paths = trie.find_all_word_paths(board)

    assert [(0, 0), (0, 1), (0, 2), (0, 3)] in paths
    assert [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)] in paths


def test_trie_deduplicates_same_word_by_lexical_path() -> None:
    board = [
        "test",
        "test",
        "abcd",
        "efgh",
    ]
    trie = Trie.build_from_words(["test"])

    paths = trie.find_all_word_paths(board)

    assert paths == [[(0, 0), (0, 1), (0, 2), (0, 3)]]


def test_trie_rejects_word_that_leaves_too_small_island() -> None:
    board = [
        "ab##",
        "cdef",
        "##gh",
        "ijkl",
    ]
    trie = Trie.build_from_words(["cdef"])

    assert trie.find_all_word_paths(board) == []


def test_leaves_small_island_returns_false_when_islands_big_enough() -> None:
    board = [
        "test",
        "abcd",
        "rate",
        "wxyz",
    ]
    removed_path = [(0, 0), (0, 1), (0, 2), (0, 3)]

    assert leaves_small_island(board, removed_path) is False


def test_leaves_small_island_treats_diagonal_chain_as_connected_when_no_trace_wall_intersects() -> None:
    board = [
        "abcd",
        "efgh",
        "ijkl",
        "mnop",
    ]
    removed_path = [
        (0, 1),
        (1, 2),
        (2, 1),
        (3, 2),
    ]

    assert leaves_small_island(board, removed_path) is False


def test_leaves_small_island_blocks_connectivity_across_removed_diagonal_trace() -> None:
    board = [
        "EEWHKC",
        "VRNAPU",
        "SPUTLG",
        "KNTITE",
    ]
    removed_path = [
        (0, 4),  # K
        (1, 5),  # U
        (2, 4),  # L
        (2, 3),  # T
        (2, 2),  # U
        (1, 1),  # R
        (2, 0),  # S
    ]

    assert leaves_small_island(board, removed_path) is True
