# from qiskit_aer.primitives import Sampler as AerSampler

# from qiskit_aqt_provider import AQTProvider
# from qiskit_aqt_provider.primitives import AQTSampler
# from qiskit_ibm_runtime.fake_provider.backends import FakeGeneva
#
# from qc_app_benchmarks.fidelity_benchmark import BenchmarkSuite
# from qc_app_benchmarks.apps import grover, qaoa, vqe, qsvm, qft, toffoli
# from qc_app_benchmarks.utils import get_fake_backend_sampler
# from qc_app_benchmarks.fidelities import normalized_fidelity
# from qc_app_benchmarks.sampler import CircuitSampler
# from qc_app_benchmarks.sampler import BosonicSampler

# ideal_sampler = CircuitSampler(AerSampler(run_options={"shots": None}))

# backend = AQTProvider("token").get_backend("offline_simulator_noise")
# aqt_sampler = AQTSampler(backend, options={"shots": 200})
# backend_sampler = CircuitSampler(get_fake_backend_sampler(FakeGeneva(), shots=1000))



#
# class BosonicSampler(BaseBenchmarkSampler):
#     def __init__(self, sampler: TBI, default_samples: int = 100):
#         super().__init__(default_samples=default_samples)
#         self.sampler = sampler
#
#     def run(self, sampler_input, num_samples=None) -> SamplerResult:
#         input_state = sampler_input[0]
#         theta_list = sampler_input[1]
#         n_samples = num_samples or self.default_samples
#         return self.sampler.sample(
#             input_state,
#             theta_list,
#             n_samples=n_samples,
#             output_format="dict",
#             n_tiling=1,
#         )


from ptseries.tbi import TBISingleLoop
from ptseries.tbi import create_tbi
from qc_app_benchmarks.apps.max_cut_orca import max_cut_thetas_6_edges

ideal_tbi = create_tbi(
    n_loops = 1
)

# ideal_sampler = BosonicSampler(ideal_tbi)

# orca_tbi = create_tbi(
#     tbi_type="PT-1",
#     n_loops = 1,
#     ip_address="169.254.109.10"
# )

orca_tbi = create_tbi(
    n_loops = 1,
    bs_loss=0.1,
    bs_noise=0.1,
    input_loss=0.1,
    detector_efficiency=0.95
)

# orca_sampler = BosonicSampler(orca_tbi, default_samples=10)


# input_state = max_cut_thetas_6_edges(return_graph=False, return_input_state=True)['input_state1']

thetas = max_cut_thetas_6_edges(return_graph=False, return_input_state=False)
input_state = (0, 0, 0, 1, 1, 1)

print(input_state)
print(thetas)

# input_state = (0, 0, 0, 1, 1, 1)
# thetas = [-1.5334, 0.0372, 0.8819, -1.9504, 0.6715, 2.6831]
# print(orca_sampler.run((input_state, thetas)))