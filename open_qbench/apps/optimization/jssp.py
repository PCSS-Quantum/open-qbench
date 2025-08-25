from qlauncher.problems import JSSP


def default_jssp() -> JSSP:
    return JSSP.from_preset("default")


def easy_jssp() -> JSSP:
    return JSSP(
        2, instance={"one": [("one", 1), ("two", 1)], "two": [("two", 1), ("one", 1)]}
    )
