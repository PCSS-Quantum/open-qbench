import numpy as np
from ptseries.tbi import create_tbi

from open_qbench.boson_sampling import (
    construct_unitary,
    generate_all_possible_outputs_orca,
    generate_analytically,
    output_probabilities,
)


def generate_dist_orca(input_state) -> dict:
    time_bin_interferometer = create_tbi()
    samples = time_bin_interferometer.sample(
        input_state=input_state,
        theta_list=[np.pi / 4] * (len(input_state) - 1),  # 50/50 beam splitters
        n_samples=1_000_000,
        output_format="dict",
    )
    tc = sum(samples.values())
    return {k: v / tc for k, v in samples.items()}


def test_analytical_generation():
    istates = [
        x
        for x in generate_all_possible_outputs_orca(
            (
                1,
                1,
                0,
            )
        )
        if sum(x) > 0
    ]
    for istate in istates:
        assert set(generate_dist_orca(istate).keys()) == (
            set(generate_analytically(istate))
        )


def test_probabilities():
    istates = [(1, 1, 2), (1, 0, 0), (1, 0, 1), (0, 0, 2)]
    for istate in istates:
        input_state = istate[::-1]
        theta_list = [np.pi / 4] * (len(input_state) - 1)
        U = construct_unitary(theta_list)
        probabilities = output_probabilities(input_state, U)

        generated = generate_dist_orca(istate)

        assert np.allclose(sum(probabilities.values()), 1.0)
        assert np.allclose(sum(generated.values()), 1.0)

        assert generated == probabilities
