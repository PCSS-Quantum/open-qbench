from collections import Counter

from qiskit.primitives import StatevectorSampler as Sampler

from open_qbench.application_benchmark import ApplicationBenchmark, FidelityAnalysis
from open_qbench.apps.circuits.ghz import ghz_decoherence_free, ghz_direct
from open_qbench.apps.circuits.grover import grover_nq
from open_qbench.apps.circuits.qaoa import jssp_7q_24d
from open_qbench.apps.circuits.qft import prepare_QFT
from open_qbench.apps.circuits.qsvm import trained_qsvm_8q
from open_qbench.apps.circuits.toffoli import toffoli_circuit
from open_qbench.apps.circuits.vqe import uccsd_3q_56d
from open_qbench.core import BenchmarkInput, BenchmarkResult
from open_qbench.metrics.fidelities import normalized_fidelity


def test_qsvm__generation():
    qc, params = trained_qsvm_8q()
    qc.measure_all()
    assert qc.num_qubits == 8
    assert len(params) == 8
    Sampler().run([(qc, params)]).result()


def test_toffoli_generation():
    qc = toffoli_circuit(5, "11111")
    qc.measure_all()
    res = Sampler().run([(qc)]).result()[0].data.meas.get_bitstrings()
    # assert res[0] == "01111"
    assert all(s == "01111" for s in res)


def test_ghz_generation():
    ghz_direct(3)
    ghz_decoherence_free(4)


def test_grover_generation():
    qc = grover_nq(4, 10)
    qc.measure_all()
    res = Sampler().run([(qc)]).result()[0].data.meas.get_counts()
    c = Counter(res)
    assert c.most_common(1)[0][0] == bin(10)[2:]


def test_qaoa_generation():
    qc, params = jssp_7q_24d()
    qc.measure_all()
    Sampler().run([(qc, params)]).result()


def test_vqe_generation():
    qc, params = uccsd_3q_56d()
    qc.measure_all()
    Sampler().run([(qc, params)]).result()


def test_qft():
    num = 24
    qc = prepare_QFT(num)
    qc.measure_all()
    res = Sampler().run([(qc)]).result()[0].data.meas.get_bitstrings()
    assert all(s == bin(num)[2:] for s in res)


def test_fidelity_analysis():
    dist_a = {
        "1101": 127,
        "0010": 108,
        "0101": 131,
        "0000": 98,
        "0100": 65,
        "0001": 90,
        "1010": 121,
        "1110": 115,
        "0111": 14,
        "1100": 30,
        "1111": 62,
        "1000": 41,
        "1011": 5,
        "1001": 2,
        "0110": 11,
        "0011": 4,
    }

    dist_b = {
        "1101": 0.1240234375,
        "0010": 0.10546875,
        "0101": 0.1279296875,
        "0000": 0.095703125,
        "0100": 0.0634765625,
        "0001": 0.087890625,
        "1010": 0.1181640625,
        "1110": 0.1123046875,
        "0111": 0.013671875,
        "1100": 0.029296875,
        "1111": 0.060546875,
        "1000": 0.0400390625,
        "1011": 0.0048828125,
        "1001": 0.001953125,
        "0110": 0.0107421875,
        "0011": 0.00390625,
    }

    res = BenchmarkResult("test", None, {}, {})
    res.execution_data["dist_ideal"] = dist_a
    res.execution_data["dist_backend"] = dist_b
    analysis = FidelityAnalysis(normalized_fidelity)
    res = analysis.run(res)
    assert res.metrics["fidelity"] == 1.0


def test_run_app_benchmark():
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit_ibm_runtime import Sampler as RSampler

    from open_qbench.apps.circuits.ghz import ghz_decoherence_free
    from open_qbench.metrics.fidelities import normalized_fidelity

    backend = GenericBackendV2(num_qubits=8)
    s = RSampler(backend)
    ss = Sampler()
    qc = ghz_decoherence_free(5)
    ben_input = BenchmarkInput(qc, s.backend())

    app_ben = ApplicationBenchmark(
        s, ss, ben_input, "GHZ", accuracy_measure=normalized_fidelity
    )
    print(app_ben)

    app_ben.run()
