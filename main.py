from qiskit_aer.primitives import Sampler as AerSampler

from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTSampler

from qc_app_benchmarks.benchmark import BenchmarkSuite
from qc_app_benchmarks.apps import grover, qaoa, vqe, qsvm, qft, toffoli


ideal_sampler = AerSampler(run_options={"shots": None})

backend = AQTProvider("token").get_backend("offline_simulator_noise")
aqt_sampler = AQTSampler(backend, options={"shots": 500})

qaoa_circuit, qaoa_params = qaoa.jssp_7q_24d()
vqe_circuit, vqe_params = vqe.uccsd_3q_56d()
qsvm_circuit, qsvm_params = qsvm.trained_qsvm_8q()
qft_circuit = qft.prepare_QFT(encoded_number=13)
grover_circuit = grover.grover_nq(4, marked_state="1010")
toffoli_circuit = toffoli.toffoli_circuit(5, input_state="11111")

suite = BenchmarkSuite(
    backend_sampler=aqt_sampler, ideal_sampler=ideal_sampler, name="test_suite"
)
suite.add_circuits(
    [
        qaoa_circuit,
        vqe_circuit,
        qsvm_circuit,
        qft_circuit,
        grover_circuit,
        toffoli_circuit,
    ],
    [qaoa_params, vqe_params, qsvm_params, [], [], []],
)
suite.run_all()
print("Results:")
for res in suite.results:
    print(
        f"{res.name} - depth = {res.normalized_depth}, fidelity = {res.average_fidelity}"
    )

suite.save_results("test_res")
suite.export_qasm("qasm_circuits")
