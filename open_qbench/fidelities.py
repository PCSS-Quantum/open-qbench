"""A file collecting different measures of fidelity between two probability distributions for use with the benchmarking suite."""

import itertools
import math

import numpy as np

from .boson_sampling import num_possible_samples


def normalized_fidelity(dist_ideal: dict, dist_backend: dict) -> float:
    """Normalized fidelity of Lubinski et al."""
    backend_fidelity = classical_fidelity(dist_ideal, dist_backend)
    uniform_fidelity = fidelity_with_uniform(dist_ideal)

    raw_fidelity = (backend_fidelity - uniform_fidelity) / (1 - uniform_fidelity)
    fidelity = max([raw_fidelity, 0])
    return fidelity


def create_normalized_fidelity(input_state: tuple):
    """Create a normalized fidelity function for a given input state -> ORCA PT"""
    def normalized_fidelity_orca(dist_ideal: dict[tuple, float], dist_backend: dict[tuple, float]) -> float:
        """Normalized fidelity modified for Boson Sampling"""
        total_photons = sum(input_state)
        num_possible = num_possible_samples(input_state)
        # filter out only 'correct' samples
        dist_ideal = {k: v for k, v in dist_ideal.items() if np.sum(k) <= total_photons}
        dist_backend = {k: v for k, v in dist_backend.items() if np.sum(k) <= total_photons}

        backend_fidelity = classical_fidelity(dist_ideal, dist_backend)
        uniform_fidelity = fidelity_with_uniform(dist_backend, num_possible)

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
    bitstrings = dist_a.keys() | dist_b.keys()
    fidelity = 0
    for b in bitstrings:
        p_a = dist_a.get(b, 0)
        p_b = dist_b.get(b, 0)
        fidelity += math.sqrt(p_a * p_b)
    fidelity = fidelity**2
    return fidelity


def fidelity_with_uniform(dist: dict, num_possible_samples: int | None = None) -> float:
    r"""Compute classical fidelity of a probability distribution with a same-sized uniform distribution

    Args:
        dist (dict): Probability distribution

    Returns:
        float: Classical fidelity given by:
        F(X,Y) = (\sum _i \sqrt{p_i q_i})^2
    """

    fidelity = 0
    if num_possible_samples is None:
        num_qubits = len(next(iter(dist.keys())))
        num_possible_samples = 2**num_qubits

    uniform_prob = 1 / num_possible_samples
    for prob in dist.values():
        fidelity += math.sqrt(prob * uniform_prob)
    fidelity = fidelity**2
    return fidelity
