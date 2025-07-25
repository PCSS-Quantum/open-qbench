import numpy as np

from examples.orca_sampler import OrcaSampler
from open_qbench.fidelities import normalized_fidelity
from open_qbench.fidelity_benchmark import (
    BenchmarkSuite,
    FidelityBenchmarkResult,
)
from open_qbench.photonics import PhotonicCircuit


def test_orca_benchmark():
    ph_circuit1 = PhotonicCircuit(input_state=[1, 1, 1, 1])
    ph_circuit1.bs(np.pi / 4, 0, 1)
    ph_circuit1.bs(np.pi / 4, 1, 2)
    ph_circuit1.bs(np.pi / 4, 2, 3)
    ph_circuit1.bs(np.pi / 4, 0, 2)
    ph_circuit1.bs(np.pi / 4, 1, 3)
    ph_circuit1.bs(np.pi / 4, 0, 3)

    ph_circuit2 = PhotonicCircuit(input_state=[1, 1, 1, 0])
    ph_circuit2.bs(np.pi / 4, 0, 1)
    ph_circuit2.bs(np.pi / 4, 0, 3)
    ph_circuit2.bs(np.pi / 4, 2, 3)
    ph_circuit2.bs(np.pi / 4, 1, 2)
    ph_circuit2.bs(np.pi / 4, 0, 2)
    ph_circuit2.bs(np.pi / 4, 1, 3)

    ideal_sampler = OrcaSampler(default_shots=1024)
    backend_sampler = OrcaSampler(default_shots=1024)

    suite = BenchmarkSuite(
        backend_sampler=backend_sampler,
        ideal_sampler=ideal_sampler,
        calculate_accuracy=normalized_fidelity,
        name="test_suite",
    )

    suite.add_benchmarks(
        [
            [
                (ph_circuit1, [np.pi / 4] * 6),
            ],
            [
                (ph_circuit2, [np.pi / 4] * 6),
            ],
        ]
    )
    suite.run_all()
    for res in suite.results:
        assert isinstance(res, FidelityBenchmarkResult)
