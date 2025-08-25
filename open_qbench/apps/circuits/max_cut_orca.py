from collections.abc import Sequence

from open_qbench.photonics.photonic_circuit import PhotonicCircuit


def _make_circuit(input_state: Sequence[int], thetas: Sequence[float]):
    return PhotonicCircuit.from_tbi_params(
        input_state=input_state, loop_lengths=[1], thetas=thetas
    )


def max_cut_3_nodes():
    return _make_circuit([1, 0, 1], [0.34181416, 1.3603507])


def max_cut_4_nodes():
    return _make_circuit([1, 0, 1, 0], [2.8409135, 2.7516844, -1.3796728])


def max_cut_5_nodes():
    return _make_circuit([1, 0, 1, 0, 1], [2.0473866, 2.179231, -2.19076, -1.1003877])


def max_cut_6_nodes():
    return _make_circuit(
        [1, 0, 1, 0, 1, 0], [2.5013795, 2.678601, 0.32164147, 1.5168, 0.05213136]
    )


def max_cut_7_nodes():
    return _make_circuit(
        [1, 0, 1, 0, 1, 0, 1],
        [-2.411401, -2.1957898, -1.4195697, 0.17244668, 0.045154937, 2.5783951],
    )


def max_cut_8_nodes():
    return _make_circuit(
        [1, 0, 1, 0, 1, 0, 1, 0],
        [
            0.98851454,
            2.5261106,
            2.608533,
            0.70516145,
            0.14947215,
            -1.6777384,
            -0.9944427,
        ],
    )


def max_cut_9_nodes():
    return _make_circuit(
        [1, 0, 1, 0, 1, 0, 1, 0, 1],
        [
            0.08544612,
            2.9686153,
            2.2086132,
            1.8140826,
            2.4022484,
            -0.4703613,
            -0.275632,
            -1.4128315,
        ],
    )
