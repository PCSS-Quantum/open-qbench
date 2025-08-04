from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import Sampler

# from qiskit_aqt_provider import AQTProvider
# from qiskit_aqt_provider.primitives import AQTSampler
from qiskit_ibm_runtime.fake_provider import FakeGeneva

from open_qbench import ApplicationBenchmark
from open_qbench.apps import grover
from open_qbench.core import BenchmarkInput
from open_qbench.fidelities import (
    normalized_fidelity,
)

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


# suite = BenchmarkSuite(
#     backend_sampler=backend_sampler,
#     ideal_sampler=ideal_sampler,
#     calculate_accuracy=normalized_fidelity,
#     name="test_suite",
# )
# suite.add_benchmarks(
#     [
#         qaoa.jssp_7q_24d(),
#         vqe.uccsd_3q_56d(),
#         qsvm.trained_qsvm_8q(),
#         qft.prepare_QFT(encoded_number=13),
#         grover.grover_nq(3, marked_state="111"),
#         toffoli.toffoli_circuit(5, input_state="11111"),
#     ]
# )
# suite.run_all()
# print("Results:")
# for res in suite.results:
#     print(
#         f"{res.name:>15}: depth = {res.input_properties['normalized_depth']}, fidelity = {res.average_fidelity}"
#     )

# suite.save_results("test_res")
# suite.export_qasm("qasm_circuits", ver=3)
