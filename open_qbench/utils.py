import json

from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_ibm_runtime import Options, QiskitRuntimeService
from qiskit_ibm_runtime import Sampler as RuntimeSampler
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

from .fidelities import normalized_fidelity


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
        backend_options={
            "method": "density_matrix",
            "coupling_map": coupling_map,
            "noise_model": noise_model,
        },
        run_options={"seed": seed, "shots": shots},
        transpile_options={"seed_transpiler": seed},
    )
    return backend_sampler


def get_ibm_backend_sampler(name: str, shots):
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend(name)
    options = Options(optimization_level=3, resilience_level=0)
    options.execution.shots = shots  # type: ignore
    ibm_sampler = RuntimeSampler(backend, options=options)

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
