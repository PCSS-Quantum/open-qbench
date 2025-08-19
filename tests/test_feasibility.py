from open_qbench.apps.optimization.jssp import default_jssp, easy_jssp
from open_qbench.metrics.feasibilities import JSSPFeasibility


def test_default_jssp():
    pr = default_jssp()
    assert JSSPFeasibility("1100110", pr)
    assert JSSPFeasibility("1100111", pr)

    assert not JSSPFeasibility("1100100", pr)

    assert not JSSPFeasibility("1110100", pr)

    assert not JSSPFeasibility("0000000", pr)


def test_easy_jssp():
    pr = easy_jssp()
    assert JSSPFeasibility("1111", pr)

    assert not JSSPFeasibility("1100", pr)

    assert not JSSPFeasibility("1010", pr)
