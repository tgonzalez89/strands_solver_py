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


def test_trie_rejects_boot_when_der_diagonal_wall_blocks_path() -> None:
    # Original board:
    #   Row 0: t o d
    #   Row 1: t e b
    #   Row 2: r o o
    #
    # DER was already played: D(0,2)->E(1,1)->R(2,0).
    # After removal those cells become '#'.
    # BOOT's only path is B(1,2)->O(2,2)->O(2,1)->T(1,0).
    # The O(2,1)->T(1,0) step crosses the ghost wall left by E(1,1)->R(2,0).
    # The solver must infer this wall from the diagonally adjacent '#' cells
    # and reject BOOT even without explicit blocked_segments from the caller.
    board_after_der = [
        "to#",
        "t#b",
        "#oo",
    ]

    trie = Trie.build_from_words(["boot"])

    paths = trie.find_all_word_paths(board_after_der)
    assert paths == [], "BOOT must be rejected because E->R ghost wall is inferred from '#' cells"


def test_trie_rejects_path_crossing_previously_played_word_diagonal_wall() -> None:
    # Original board:
    #   Row 0: H T R A E H
    #   Row 1: Q R F R S I
    #   Row 2: U A W I E R
    #   Row 3: E S T D N G
    #
    # DWARF was played: D(3,3)->W(2,2)->A(2,1)->R(1,1)->F(1,2).
    # After removal those cells become '#'.
    # STIR path S(3,1)->T(3,2)->I(2,3)->R(1,3) is the only possible path,
    # but the T->I step crosses the ghost diagonal wall left by D->W.
    # Without blocked_segments the path is found; with them it must be rejected.
    board_after_dwarf = [
        "htraeh",
        "q##rsi",
        "u##ier",
        "est#ng",
    ]

    trie = Trie.build_from_words(["stir"])

    # STIR must be rejected: the '#' cells left by DWARF have diagonally
    # adjacent pairs that imply the D->W wall; T->I crosses it.
    paths = trie.find_all_word_paths(board_after_dwarf)
    assert paths == [], "STIR must be rejected because the D->W ghost wall is inferred from '#' cells"


def test_trie_finds_word_on_late_game_board_without_overblocking() -> None:
    board = [
        "eolbig",
        "rvrseg",
        "yust##",
        "otf###",
        "a##a##",
        "n###n#",
        "######",
        "######",
    ]
    trie = Trie.build_from_words(["gets"])

    paths = trie.find_all_word_paths(board)

    assert paths == [[(1, 5), (1, 4), (2, 3), (1, 3)]]
