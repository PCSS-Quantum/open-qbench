import numpy as np

from open_qbench.application_benchmark import (
    ApplicationBenchmark,
    # BenchmarkSuite,
)
from open_qbench.core import BenchmarkInput
from open_qbench.metrics.fidelities import classical_fidelity
from open_qbench.orca.sampler import OrcaSampler
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

    ben_input = BenchmarkInput(ph_circuit1, None)
    orca_ben = ApplicationBenchmark(
        ideal_sampler,
        ideal_sampler,
        ben_input,
        name="test",
        accuracy_measure=classical_fidelity,
    )
    orca_ben.run()
    print(orca_ben.result)

    # suite = BenchmarkSuite(
    #     backend_sampler=backend_sampler,
    #     ideal_sampler=ideal_sampler,
    #     calculate_accuracy=normalized_fidelity,
    #     name="test_suite",
    # )

    # suite.add_benchmarks(
    #     [
    #         [
    #             (ph_circuit1, [np.pi / 4] * 6),
    #         ],
    #         [
    #             (ph_circuit2, [np.pi / 4] * 6),
    #         ],
    #     ]
    # )
    # suite.run_all()
    # for res in suite.results:
    #     assert isinstance(res, ApplicationBenchmarkResult)
