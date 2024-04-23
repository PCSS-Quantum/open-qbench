"""
This file collects different measures of fidelity between two probability distributions
for use with the benchmarking suite.
"""

import itertools
import math
import numpy as np
from ptseries.tbi import create_tbi



def normalized_fidelity(dist_ideal: dict, dist_backend: dict) -> float:
    """Normalized fidelity of Lubinski et al."""
    backend_fidelity = classical_fidelity(dist_ideal, dist_backend)
    num_qubits = len(list(dist_ideal.keys())[0])
    uniform_fidelity = classical_fidelity(dist_ideal, _uniform_dist(num_qubits))

    raw_fidelity = (backend_fidelity - uniform_fidelity) / (1 - uniform_fidelity)

    fidelity = max([raw_fidelity, 0])
    return fidelity


def normalized_fidelity_orca(dist_ideal: dict, dist_backend: dict) -> float:
    """Normalized fidelity modified for Boson Sampling"""
    backend_fidelity = classical_fidelity(dist_ideal, dist_backend)
    input_state = (1, 0, 1, 0, 1, 0)
    uniform_fidelity = classical_fidelity(dist_ideal, _uniform_dist_orca(input_state))

    raw_fidelity = (backend_fidelity - uniform_fidelity) / (1 - uniform_fidelity)

    fidelity = max([raw_fidelity, 0])
    return fidelity

def classical_fidelity(dist_a: dict, dist_b: dict) -> float:
    """Compute classical fidelity of two probability distributions

    Args:
        counts_a (dict): Distribution of experiment A
        counts_b (dict): Distribution of experiment B

    Returns:
        float: Classical fidelity
    """
    if False:
        num_qubits = len(list(dist_a.keys())[0])
        bitstrings = ("".join(i) for i in itertools.product("01", repeat=num_qubits))
    else:
        num_qubits = len(list(dist_a.keys())[0])
        bitstrings = generate_all_possible_outputs_orca((1,) * num_qubits)
    fidelity = 0
    for b in bitstrings:
        p_a = dist_a.get(b, 0)
        p_b = dist_b.get(b, 0)
        fidelity += math.sqrt(p_a * p_b)
    fidelity = fidelity**2
    return fidelity


def _uniform_dist(num_qubits) -> dict:
    bitstrings = ("".join(i) for i in itertools.product("01", repeat=num_qubits))
    prob = 1 / num_qubits**2
    dist = {b: prob for b in bitstrings}
    return dist

def _uniform_dist_orca(input_state) -> dict:
    samples = generate_all_samples_orca(input_state)
    prob = 1 / len(samples)
    dist = {s: prob for s in samples}
    return dist


def counts_to_dist(counts: dict) -> dict:
    shots = sum(counts.values())
    dist = {x: y / shots for x, y in counts.items()}
    return dist

def generate_all_samples_orca(input_state):
    time_bin_interferometer = create_tbi()
    samples = time_bin_interferometer.sample(
    input_state=input_state,
    theta_list=[np.pi/4] * (len(input_state) - 1), # 50/50 beam splitters
    n_samples=100000)
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

if __name__ == "__main__":
    ans1 = generate_all_samples_orca((1, 0, 1))
    ans2 = generate_all_possible_outputs_orca((1, 0, 1))
    print(ans1)
    print(len(ans1))
    print(ans2)
    print(len(ans2))
