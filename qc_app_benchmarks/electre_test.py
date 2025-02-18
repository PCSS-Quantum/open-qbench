from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTSampler
from qiskit_ibm_runtime.fake_provider.backends import FakeGeneva
from qc_app_benchmarks.apps import grover, qaoa, qft, qsvm, toffoli, vqe
from qc_app_benchmarks.fidelities import normalized_fidelity
from qc_app_benchmarks.sampler.circuit_sampler import CircuitSampler
from qc_app_benchmarks.utils import get_fake_backend_sampler
from qc_app_benchmarks.backend_comparison import backend_comparison


def test_electre():
    ideal_sampler = CircuitSampler(AerSampler(run_options={"shots": None}))

    backend = AQTProvider("token").get_backend("offline_simulator_noise")
    aqt_sampler = AQTSampler(backend)
    aqt_sampler = CircuitSampler(aqt_sampler, default_samples=200)
    fake_sampler = CircuitSampler(get_fake_backend_sampler(FakeGeneva()), default_samples=1000)

    backend_samplers = {"aqt": aqt_sampler, "fake": fake_sampler}

    benchmark_inputs = [
        grover.grover_nq(3, marked_state="111"),
        qaoa.jssp_7q_24d(),
        qft.prepare_QFT(encoded_number=13),
        qsvm.trained_qsvm_8q(),
        toffoli.toffoli_circuit(5, input_state="11111"),
        vqe.uccsd_3q_56d(),
    ]

    Q = [0.02] * 6
    P = [0.1] * 6
    V = [0.4] * 6
    W = [1] * 6

    global_concordance, credibility, rank_D, rank_A, rank_N, rank_P = backend_comparison(
        backend_samplers, ideal_sampler, normalized_fidelity, benchmark_inputs, Q, P, V, W, "graph.png"
    )
    assert global_concordance is not None
    assert credibility is not None
    assert len(rank_D) == len(backend_samplers)
    assert len(rank_A) == len(backend_samplers)
    assert len(rank_N) == len(backend_samplers)
    assert len(rank_P) == len(backend_samplers)
    
    
