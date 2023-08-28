from qiskit.providers.fake_provider import FakeHanoi

from qiskit_aer.primitives import Sampler as AerSampler

from qiskit_ibm_runtime import Sampler as RuntimeSampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Options

from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTSampler

from qc_app_benchmarks.benchmark import BenchmarkSuite
from qc_app_benchmarks.apps import qaoa, vqe, qsvm, qft
from qc_app_benchmarks.utils import create_fake_backend_sampler


ideal_sampler = AerSampler(backend_options={"method": "statevector"})

backend = AQTProvider("token").get_backend("offline_simulator_noise")
aqt_sampler = AQTSampler(backend, options={"shots": 500})

backend_sampler = create_fake_backend_sampler(FakeHanoi(), shots=1000, seed=123)

# service = QiskitRuntimeService(channel="ibm_quantum")
# backend = service.backend("ibm_sherbrooke")
# options = Options(optimization_level=3, resilience_level=0)
# options.execution.shots = 500
# ibm_sampler = RuntimeSampler(backend, options=options)

qaoa_circuit, qaoa_params = qaoa.jssp_7q_24d()
vqe_circuit, vqe_params = vqe.uccsd_3q_56d()
qsvm_circuit, qsvm_params = qsvm.trained_qsvm_8q()
qft_circuit = qft.prepare_QFT(encoded_number=13)

suite = BenchmarkSuite(
    backend_sampler=aqt_sampler, ideal_sampler=ideal_sampler, name="test_suite"
)
suite.add_circuits(
    [qaoa_circuit, vqe_circuit, qsvm_circuit, qft_circuit],
    [qaoa_params, vqe_params, qsvm_params, None],
)
suite.run_all()
suite.calculate_fidelities()
for ben in suite:
    print(
        f"{ben.name} depth: {ben.normalized_depth()}, fidelity: {ben.result.average_fidelity}"
    )
suite.save_results("test_res")
suite.export_qasm("qasm_circuits")
