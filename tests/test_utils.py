from open_qbench.utils import check_tuple_types


def test_tuple_types():
    assert check_tuple_types((1, "2"), [int, str])
    assert not check_tuple_types((1, "2"), [int, int])
    assert check_tuple_types((1, (1, "2")), [int, tuple])
    assert not check_tuple_types((1, (1, "2")), [int, list])
    assert check_tuple_types((1, (1, "2")), [int, [int, str]], recursive=True)
    assert not check_tuple_types((1, (1, "2")), [int, [int, int]], recursive=True)
