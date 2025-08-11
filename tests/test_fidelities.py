"""
This file collects different measures of fidelity between two probability distributions
for use with the benchmarking suite.
"""

import itertools
import math
import time

import numpy as np
from ptseries.tbi import create_tbi

from open_qbench.fidelities import (
    normalized_fidelity as normalized_fidelity_dev,
    create_normalized_fidelity as create_normalized_fidelity_dev,
    classical_fidelity as classical_fidelity_dev,
    fidelity_with_uniform as fidelity_with_uniform_dev,
)

from open_qbench.boson_sampling import num_possible_samples


# -------------------------------------------
# Old functions from fidelities.py
# -------------------------------------------

def normalized_fidelity(dist_ideal: dict, dist_backend: dict) -> float:
    """Normalized fidelity of Lubinski et al."""
    backend_fidelity = classical_fidelity(dist_ideal, dist_backend)
    uniform_fidelity = fidelity_with_uniform(dist_ideal)

    raw_fidelity = (backend_fidelity - uniform_fidelity) / (1 - uniform_fidelity)
    fidelity = max([raw_fidelity, 0])
    return fidelity


def create_normalized_fidelity(input_state):
    def normalized_fidelity_orca(dist_ideal: dict, dist_backend: dict) -> float:
        """Normalized fidelity modified for Boson Sampling"""
        backend_fidelity = classical_fidelity_orca(
            dist_ideal, dist_backend, input_state
        )
        uniform_fidelity = classical_fidelity_orca(
            dist_ideal, _uniform_dist_orca(input_state), input_state
        )

        raw_fidelity = (backend_fidelity - uniform_fidelity) / (1 - uniform_fidelity)

        fidelity = max([raw_fidelity, 0])
        return fidelity

    return normalized_fidelity_orca


def classical_fidelity(dist_a: dict, dist_b: dict) -> float:
    r"""Compute classical fidelity of two probability distributions

    Args:
        dist_a (dict): Distribution of experiment A
        dist_b (dict): Distribution of experiment B

    Returns:
        float: Classical fidelity given by:
        F(X,Y) = (\sum _i \sqrt{p_i q_i})^2
    """
    num_qubits = len(next(iter(dist_a.keys())))
    bitstrings = ("".join(i) for i in itertools.product("01", repeat=num_qubits))
    fidelity = 0
    for b in bitstrings:
        p_a = dist_a.get(b, 0)
        p_b = dist_b.get(b, 0)
        fidelity += math.sqrt(p_a * p_b)
    fidelity = fidelity**2
    return fidelity


def classical_fidelity_orca(
    dist_a: dict, dist_b: dict, input_state: list[int]
) -> float:
    r"""Compute classical fidelity of two probability distributions

    Args:
        dist_a (dict): Distribution of experiment A
        dist_b (dict): Distribution of experiment B

    Returns:
        float: Classical fidelity given by:
        F(X,Y) = (\sum _i \sqrt{p_i q_i})^2
    """
    bitstrings = generate_all_possible_outputs_orca(input_state)
    fidelity = 0
    for b in bitstrings:
        p_a = dist_a.get(b, 0)
        p_b = dist_b.get(b, 0)
        fidelity += math.sqrt(p_a * p_b)
    fidelity = fidelity**2
    return fidelity


def fidelity_with_uniform(dist: dict) -> float:
    r"""Compute classical fidelity of a probability distribution with a same-sized uniform distribution

    Args:
        dist (dict): Probability distribution

    Returns:
        float: Classical fidelity given by:
        F(X,Y) = (\sum _i \sqrt{p_i q_i})^2
    """
    num_qubits = len(next(iter(dist.keys())))
    fidelity = 0
    uniform_prob = 1 / 2**num_qubits
    for prob in dist.values():
        fidelity += math.sqrt(prob * uniform_prob)
    fidelity = fidelity**2
    return fidelity


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
        theta_list=[np.pi / 4] * (len(input_state) - 1),  # 50/50 beam splitters
        n_samples=1000000,
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

# -------------------------------------------
# / Old functions from fidelities.py
# -------------------------------------------

# -------------------------------------------
# helpers
# -------------------------------------------


def generate_random_dist_bits(n_qubits=4):
    bitstrings = ("".join(i) for i in itertools.product("01", repeat=n_qubits))
    dist = np.random.random_integers(0, 100, size=(2**n_qubits,))
    dist = dist/np.sum(dist)
    return dict(zip(bitstrings, dist))


def generate_random_dist_orca(input_state):
    outputs = generate_all_samples_orca(input_state)

    dist = np.random.random_integers(0, 100, size=(len(outputs),))
    dist = dist/np.sum(dist)
    return dict(zip(outputs, dist))

# -------------------------------------------
# / helpers
# -------------------------------------------


def test_classical_fidelity():
    """Test if classical fidelity results match"""
    d1, d2 = generate_random_dist_bits(), generate_random_dist_bits()
    assert np.allclose(classical_fidelity(d1, d2), classical_fidelity_dev(d1, d2))


def test_normalized_fidelity():
    """Test if normalized fidelity results match"""
    d1, d2 = generate_random_dist_bits(), generate_random_dist_bits()
    assert np.allclose(normalized_fidelity(d1, d2), normalized_fidelity(d1, d2))


def test_classical_fidelity_orca():
    """Test if classical fidelity results match for orca distributions"""
    istate = [1, 0, 1, 0, 1, 0]
    d1, d2 = generate_random_dist_orca(istate), generate_random_dist_orca(istate)
    assert np.allclose(classical_fidelity_dev(d1, d2), classical_fidelity_orca(d1, d2, istate))


def test_uniform_orca():
    """Test if fidelity with uniform results match for orca distributions"""
    istate = [1, 0, 1, 0, 1, 0]
    dist = generate_random_dist_orca(istate)
    assert np.allclose(
        fidelity_with_uniform_dev(dist, num_possible_samples=num_possible_samples(istate)),
        classical_fidelity_orca(dist, _uniform_dist_orca(istate), istate)
    )


def test_samples():
    """
    Test if calculation of the number of possible samples matches 
    between the ptseries tbi approach and our analytical approach
    """
    istate = [1, 0, 1, 0, 1, 0]
    s1 = time.time()
    new = num_possible_samples(istate)
    l1 = time.time() - s1

    s2 = time.time()
    old = len(generate_all_samples_orca(istate))
    l2 = time.time() - s2

    assert l1 <= l2
    assert old == new


def test_normalized_fidelity_orca():
    """Test if normalized fidelity results match for orca distributions"""
    istate = [1, 0, 1, 0, 1, 0]
    d1, d2 = generate_random_dist_orca(istate), generate_random_dist_orca(istate)
    old, new = create_normalized_fidelity(istate)(d1, d2), create_normalized_fidelity_dev(istate)(d1, d2)
    assert np.allclose(old, new)
