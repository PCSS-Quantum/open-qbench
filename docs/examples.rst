Examples
========

IBM Fidelity
--------------

::

    from qiskit_aer.primitives import SamplerV2 as AerSampler

    from qiskit_ibm_runtime.fake_provider.backends import FakeGeneva

    from qc_app_benchmarks.apps import grover, qaoa, qft, qsvm, toffoli, vqe
    from qc_app_benchmarks.fidelities import classical_fidelity, normalized_fidelity
    from qc_app_benchmarks.fidelity_benchmark import BenchmarkSuite, FidelityBenchmark
    from qc_app_benchmarks.sampler.circuit_sampler import CircuitSampler
    from qc_app_benchmarks.utils import calculate_from_file, get_fake_backend_sampler

    ideal_sampler = CircuitSampler(AerSampler())

    backend_sampler = CircuitSampler(
        get_fake_backend_sampler(FakeGeneva()), default_samples=1000
    )

    suite = BenchmarkSuite(
        backend_sampler=backend_sampler,
        ideal_sampler=ideal_sampler,
        calculate_accuracy=normalized_fidelity,
        name="test_suite",
    )
    
    suite.add_benchmarks(
        [
            qaoa.jssp_7q_24d(),
            vqe.uccsd_3q_56d(),
            qsvm.trained_qsvm_8q(),
            qft.prepare_QFT(encoded_number=13),
            grover.grover_nq(3, marked_state="111"),
            toffoli.toffoli_circuit(5, input_state="11111"),
        ]
    )
    suite.run_all()
    print("Results:")
    for res in suite.results:
        print(
            f"{res.name:>15}: depth = {res.input_properties['normalized_depth']}, fidelity = {res.average_fidelity}"
        )

    suite.save_results("test_res")
    suite.export_qasm("qasm_circuits", ver=3)


ORCA Fidelity
--------------

::

    """Running this example requires adding your SSH key to https://sdk.orcacomputing.com/ and installing with pip install .[ORCA]"""

    from qc_app_benchmarks.photonics import PhotonicCircuit
    from examples.orca_sampler import OrcaSampler
    from qc_app_benchmarks.fidelities import create_normalized_fidelity, normalized_fidelity
    from qc_app_benchmarks.fidelity_benchmark import BenchmarkSuite
    import dill

    ph_circuit1 = PhotonicCircuit.from_tbi_params([1,0,1,0,1,0],[1],[0.8479, -0.0095, 0.2154, -1.3921, 0.0614])


    ideal_sampler = OrcaSampler(default_shots=1024)
    backend_sampler = OrcaSampler(default_shots=1024)

    normalized_fidelity_orca = create_normalized_fidelity(input_state=[1,0,1,0,1,0])

    suite = BenchmarkSuite(
        backend_sampler=backend_sampler,
        ideal_sampler=ideal_sampler,
        calculate_accuracy=normalized_fidelity_orca,
        name="test_suite",
    )
    suite.add_benchmarks(
        [
            [(ph_circuit1, [0.8479, -0.0095, 0.2154, -1.3921, 0.0614]),],
        ]
    )
    suite.run_all()
    print("Results:")
    for res in suite.results:
        print(
            f"{res.name:>15}: depth = {res.input_properties['normalized_depth']}, fidelity = {res.average_fidelity}"
        )
    suite.save_results("test_res")

    with open("object_dump","wb") as f:
        dill.dump(suite, f)
