"""Functions related to analytical generation of boson sampling results"""

import itertools
import math

import numpy as np
from ptseries.tbi import create_tbi


def generate_all_samples_orca(input_state):
    """Generate all possible samples via ptseries tbi sampling"""
    time_bin_interferometer = create_tbi()
    samples = time_bin_interferometer.sample(
        input_state=input_state,
        theta_list=[np.pi / 4] * (len(input_state) - 1),  # 50/50 beam splitters
        n_samples=100000,
    )
    samples_sorted = dict(sorted(samples.items(), key=lambda state: -state[1]))
    labels = list(samples_sorted.keys())
    return labels


def generate_all_possible_outputs_orca(input_state):
    all_possible_outputs = []
    max_num_photons = np.sum(input_state)
    # print(max_num_photons)
    tab_num_photons = range(0, max_num_photons + 1)
    arr = itertools.product(tab_num_photons, repeat=len(input_state))

    for el in arr:
        if np.sum(el) <= np.sum(input_state):
            all_possible_outputs.append(el)
    return all_possible_outputs


def per(mtx, column, selected, prod, output=False):
    if column == mtx.shape[1]:
        if output:
            print(selected, prod)
        return prod
    else:
        result = 0
        for row in range(mtx.shape[0]):
            if row not in selected:
                result = result + per(
                    mtx,
                    column + 1,
                    [*selected, row],
                    prod * mtx[row, column],
                )
        return result


def compute_permanent(mat):
    return per(mat, 0, [], 1)


def construct_unitary(thetas):
    """Construct a unitary of a TBI with given beam splitter angles"""

    def beamsplitter_matrix(theta, n, i, j):
        U = np.eye(n, dtype=complex)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        U[i, i] = cos_theta
        U[i, j] = -sin_theta
        U[j, i] = sin_theta
        U[j, j] = cos_theta

        return U

    def apply_sequential_beam_splitters(thetas):
        n = len(thetas) + 1
        U = np.eye(n, dtype=complex)
        for k, theta in enumerate(thetas):
            if k + 1 < n:
                BS = beamsplitter_matrix(theta, n, k, k + 1)
                U = BS @ U
        return U

    return apply_sequential_beam_splitters(thetas)


def generate_submatrix(U, input_bitstring, output):
    columns = [j for j, t in enumerate(output) for _ in range(t)]
    UT = U[:, columns]

    rows = [i for i, s in enumerate(input_bitstring) for _ in range(s)]
    UST = UT[rows, :]
    return UST


def output_probabilities(input_state: tuple, U: np.ndarray) -> dict[tuple, float]:
    """Calculate sample probabilities for a given input state and TBI unitary"""
    possible_outputs = generate_all_possible_outputs_orca(input_state)
    probabilities = {}

    for output_indices in possible_outputs:
        U_submatrix = generate_submatrix(U, input_state, output_indices)
        PQT = np.abs(compute_permanent(U_submatrix)) ** 2
        probabilities[tuple(output_indices)] = PQT / (
            np.prod([math.factorial(s) for s in input_state])
            * np.prod([math.factorial(t) for t in output_indices])
        )

    total_prob = sum(probabilities.values())
    for key in probabilities:
        probabilities[key] /= total_prob

    return probabilities


def generate_analytically(input_state: tuple) -> list[tuple]:
    """Generate possible samples for a given input state"""
    input_state = input_state[::-1]
    theta_list = [np.pi / 4] * (len(input_state) - 1)
    U = construct_unitary(theta_list)
    probabilities = output_probabilities(input_state, U)
    return [
        (key[::-1])
        for key, value in probabilities.items()
        if value > 1e-30 and sum(key) == sum(input_state)
    ]


def num_possible_samples(input_state: tuple) -> int:
    """Number of possible samples for a given input state"""
    return len(generate_analytically(input_state))


if __name__ == "__main__":
    input_string = (1, 1, 2)
    ans0 = generate_analytically(input_string)
    ans1 = generate_all_samples_orca(input_string)
    ans2 = generate_all_possible_outputs_orca(input_string)
    print(ans0)
    print(len(ans0))
    print(ans1)
    print(len(ans1))
    print(ans2)
    print(len(ans2))
