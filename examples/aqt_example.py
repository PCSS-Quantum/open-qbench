from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTSampler

from open_qbench import ApplicationBenchmark
from open_qbench.analysis import FidelityAnalysis
from open_qbench.apps.circuits import (
    ghz_decoherence_free,
    grover_nq,
    prepare_QFT,
    toffoli_circuit,
    uccsd_3q_56d,
)
from open_qbench.core import BenchmarkInput
from open_qbench.metrics.fidelities import normalized_fidelity

provider = AQTProvider("ACCESS_TOKEN")
# direct_access_backend = provider.get_direct_access_backend(
#     "http://192.168.34.4/api/v1/"
# )
# print(provider.backends())
backend = provider.get_backend("offline_simulator_noise")
backend.configuration().max_shots = 200  # this is how the DA API is configured

ideal_sampler = AerSampler()
backend_sampler = AQTSampler(backend)
backend_sampler.set_transpile_options(optimization_level=3)

circuits = [
    ghz_decoherence_free(6),
    grover_nq(3, 6),
    prepare_QFT(encoded_number=11),
    toffoli_circuit(4, "1111"),
    uccsd_3q_56d(),
]
options = {"backend_shots": 1000, "simulator_shots": 1000}
benchmarks = []

for circ in circuits:
    benchmark_input = BenchmarkInput(
        circ,
        backend_sampler.backend,
        options,
    )
    ab = ApplicationBenchmark(
        backend_sampler,
        ideal_sampler,
        benchmark_input,
        analysis=FidelityAnalysis(normalized_fidelity),
        name=circ.name,
    )
    benchmarks.append(ab)
    ab.run()
    print(ab.result)

# manager = BenchmarkManager(*benchmarks)
# manager.run_all()
# manager.save_results()
