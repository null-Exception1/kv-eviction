from src.index_map import IndexMap


def test_compact_reindexes_global_positions_to_contiguous_logical_positions():
    index_map = IndexMap()
    mapping = index_map.compact([505, 100, 300])

    assert mapping == {100: 0, 300: 1, 505: 2}
    assert index_map.logical_position(300) == 1
