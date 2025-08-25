import json
from collections.abc import Sequence

from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
from qiskit_ibm_runtime.fake_provider.fake_backend import (
    FakeBackendV2,
)

from .metrics.fidelities import normalized_fidelity


def get_fake_backend_sampler(
    fake_backend: FakeBackendV2,
    shots: int | None = None,
    seed: int | None = None,
) -> AerSampler:
    """Creates a sampler from qiskit_aer based on a noise model supplied by a Qiskit fake backend

    Args:
        fake_backend (FakeBackendV2): an object representing a Qiskit fake backend
        shots (int): number of shots for the sampler
        seed (Optional[int], optional): Random seed for the simulator and the transpiler.
        Defaults to None.

    Returns:
        AerSampler: _description_
    """
    coupling_map = fake_backend.coupling_map
    noise_model = NoiseModel.from_backend(fake_backend)

    backend_sampler = AerSampler(
        options={
            "backend_options": {
                "method": "density_matrix",
                "coupling_map": coupling_map,
                "noise_model": noise_model,
            },
            "run_options": {"seed": seed, "shots": shots},
            # TODO: find another way to parametrize transpilation
            # 'transpile_options':
            #     {
            #         "seed_transpiler": seed
            #     },
        }
    )
    return backend_sampler


def get_ibm_backend_sampler(name: str, shots):
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend(name)
    # TODO: no transpilation options available for Sampler v2
    # options = Options(optimization_level=3, resilience_level=0)
    ibm_sampler = RuntimeSampler(
        backend,
        options={
            "default_shots": shots,
        },
    )
    return ibm_sampler


def calculate_from_file(file: str) -> float:
    """Recalculate the normalized fidelity from a JSON file with benchmark results

    Args:
        file (str): A path to a JSON file with benchmark results

    Returns:
        float: Normalized fidelity of the provided distributions
    """
    with open(file, "rb") as f:
        result = json.load(f)
    return normalized_fidelity(result["dist_ideal"], result["dist_backend"])


def check_tuple_types(
    var: tuple, types: Sequence[type | Sequence[type]], recursive: bool = False
) -> bool:
    """

    Check if tuple contains declared types.
    Think isinstance(tup, tuple[Type1,Type2])

    Args:
        var (tuple): Tuple to be checked
        types (list[type]): Desired types, in order.
        recursive (bool, optional):
            If element i in tuple is a tuple and element i in types is a list, decide whether to check it as well. Defaults to False.

    Returns:
        bool: Whether the check is successful
    """

    if not isinstance(var, tuple):
        return False

    if len(var) != len(types):
        return False

    for t_val, val_type in zip(var, types, strict=False):
        if not isinstance(val_type, Sequence):
            if not isinstance(t_val, val_type):
                return False
        else:
            if (
                isinstance(t_val, tuple)
                and recursive
                and not check_tuple_types(t_val, val_type)
            ):
                return False

            if not isinstance(t_val, tuple):
                return False
    return True
