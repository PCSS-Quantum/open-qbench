from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.providers.fake_provider import FakeBackendV2, FakePulseBackend


def create_fake_backend_sampler(
    fake_backend: FakeBackendV2 | FakePulseBackend, shots: int, seed: int = None
) -> AerSampler:
    backend = fake_backend
    coupling_map = backend.configuration().coupling_map
    noise_model = NoiseModel.from_backend(backend)

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
