Examples
========

IBM Fidelity
--------------

.. code-block:: python

    from qiskit_aer.primitives import SamplerV2 as AerSampler
    from qiskit_ibm_runtime import Sampler
    from qiskit_ibm_runtime.fake_provider import FakeGeneva

    from open_qbench import ApplicationBenchmark
    from open_qbench.apps import grover
    from open_qbench.core import BenchmarkInput
    from open_qbench.fidelities import normalized_fidelity

    ideal_sampler = AerSampler(default_shots=1000)
    backend_sampler = Sampler(FakeGeneva())

    ab = ApplicationBenchmark(
        backend_sampler,
        ideal_sampler,
        BenchmarkInput(grover.grover_nq(3, 6), backend_sampler.backend()),
        name="Grover_benchmark",
        accuracy_measure=normalized_fidelity,
    )

    ab.run()
    print(ab.result)


ORCA Fidelity
--------------

.. code-block:: python

    from functools import partial

    from orca_sampler import OrcaSampler

    from open_qbench.application_benchmark import ApplicationBenchmark
    from open_qbench.core import BenchmarkInput
    from open_qbench.fidelities import classical_fidelity_orca
    from open_qbench.photonics import PhotonicCircuit

    input_state = [1, 0, 1, 0, 1, 0]
    ph_circuit1 = PhotonicCircuit.from_tbi_params(
        input_state=input_state,
        loop_lengths=[1],
        thetas=[0.8479, -0.0095, 0.2154, -1.3921, 0.0614],
    )
    fidelity = partial(classical_fidelity_orca, input_state=input_state)

    ideal_sampler = OrcaSampler(default_shots=1024)

    ben_input = BenchmarkInput(ph_circuit1)
    orca_ben = ApplicationBenchmark(
        ideal_sampler,
        ideal_sampler,
        ben_input,
        name="ORCA_fidelity",
        accuracy_measure=fidelity,
    )
    print(orca_ben.benchmark_input)
    orca_ben.run()
    print(orca_ben.result)
