"""
This file collects different measures of fidelity between two probability distributions
for use with the benchmarking suite.
"""

import itertools
import math


def normalized_fidelity(dist_ideal: dict, dist_backend: dict) -> float:
    """Normalized fidelity of Lubinski et al."""
    backend_fidelity = classical_fidelity(dist_ideal, dist_backend)
    uniform_fidelity = fidelity_with_uniform(dist_ideal)

    raw_fidelity = (backend_fidelity - uniform_fidelity) / (1 - uniform_fidelity)
    fidelity = max([raw_fidelity, 0])
    return fidelity


def classical_fidelity(dist_a: dict, dist_b: dict) -> float:
    """Compute classical fidelity of two probability distributions

    Args:
        dist_a (dict): Distribution of experiment A
        dist_b (dict): Distribution of experiment B

    Returns:
        float: Classical fidelity given by:
        F(X,Y) = (\sum _i \sqrt{p_i q_i})^2
    """
    num_qubits = len(list(dist_a.keys())[0])
    bitstrings = ("".join(i) for i in itertools.product("01", repeat=num_qubits))
    fidelity = 0
    for b in bitstrings:
        p_a = dist_a.get(b, 0)
        p_b = dist_b.get(b, 0)
        fidelity += math.sqrt(p_a * p_b)
    fidelity = fidelity**2
    return fidelity


def fidelity_with_uniform(dist: dict) -> float:
    """Compute classical fidelity of a probability distribution with a same-sized uniform distribution

    Args:
        dist (dict): Probability distribution

    Returns:
        float: Classical fidelity given by:
        F(X,Y) = (\sum _i \sqrt{p_i q_i})^2
    """
    num_qubits = len(list(dist.keys())[0])
    fidelity = 0
    uniform_prob = 1 / 2**num_qubits
    for prob in dist.values():
        fidelity += math.sqrt(prob * uniform_prob)
    fidelity = fidelity**2
    return fidelity


def counts_to_dist(counts: dict) -> dict:
    shots = sum(counts.values())
    dist = {x: y / shots for x, y in counts.items()}
    return dist
