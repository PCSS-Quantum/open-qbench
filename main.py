from qiskit_aer.primitives import Sampler as AerSampler

from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTSampler
from qiskit_ibm_runtime.fake_provider.backends import FakeGeneva

from qc_app_benchmarks.fidelity_benchmark import BenchmarkSuite, FidelityBenchmark
from qc_app_benchmarks.apps import grover, qaoa, vqe, qsvm, qft, toffoli
from qc_app_benchmarks.utils import get_fake_backend_sampler
from qc_app_benchmarks.fidelities import normalized_fidelity, classical_fidelity
from qc_app_benchmarks.sampler.circuit_sampler import CircuitSampler

ideal_sampler = CircuitSampler(AerSampler(run_options={"shots": None}))


backend = AQTProvider("token").get_backend("offline_simulator_noise")
aqt_sampler = AQTSampler(backend)
backend_sampler = CircuitSampler(aqt_sampler, default_samples=200)
# backend_sampler = CircuitSampler(
#     get_fake_backend_sampler(FakeGeneva()), default_samples=1000
# )

# fb = FidelityBenchmark(backend_sampler, ideal_sampler, qaoa.jssp_7q_24d(), "test")
# fb.calculate_accuracy = classical_fidelity
# res = fb.run()
# print(f"{res=}")

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

# suite.save_results("test_res")
# suite.export_qasm("qasm_circuits", ver=3)
