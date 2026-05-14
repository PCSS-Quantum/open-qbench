from collections import Counter

from qiskit.primitives import StatevectorSampler as Sampler

from open_qbench.analysis import FidelityAnalysis
from open_qbench.apps.circuits import (
    ghz_decoherence_free,
    ghz_direct,
    grover_nq,
    jssp_7q_24d,
    prepare_QFT,
    toffoli_circuit,
    trained_qsvm_8q,
    uccsd_3q_56d,
)
from open_qbench.benchmarks import ApplicationBenchmark, OptimizationBenchmark
from open_qbench.core import BenchmarkInput
from open_qbench.metrics.fidelities import normalized_fidelity


def test_qsvm__generation():
    qc = trained_qsvm_8q()
    assert qc.num_qubits == 8
    Sampler().run([(qc)]).result()


def test_toffoli_generation():
    qc = toffoli_circuit(5, "11111")
    data = Sampler().run([(qc)]).result()[0].data
    register_name = next(iter(vars(data).keys()))
    bitstrings = getattr(data, register_name).get_bitstrings()
    # assert res[0] == "01111"
    assert all(s == "01111" for s in bitstrings)


def test_ghz_generation():
    ghz_direct(3)
    ghz_decoherence_free(4)


def test_grover_generation():
    qc = grover_nq(4, 10)
    res = Sampler().run([(qc)]).result()[0].data.meas.get_counts()
    c = Counter(res)
    assert c.most_common(1)[0][0] == bin(10)[2:]


def test_qaoa_generation():
    qc = jssp_7q_24d()
    Sampler().run([(qc)]).result()


def test_vqe_generation():
    qc = uccsd_3q_56d()
    Sampler().run([(qc)]).result()


def test_qft():
    num = 24
    qc = prepare_QFT(num)
    res = Sampler().run([(qc)]).result()[0].data.meas.get_bitstrings()
    assert all(s == bin(num)[2:] for s in res)


def test_run_app_benchmark():
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit_ibm_runtime import Sampler as RSampler

    from open_qbench.apps.circuits.ghz import ghz_decoherence_free

    backend = GenericBackendV2(num_qubits=8)
    s = RSampler(backend)
    ss = Sampler()
    qc = ghz_decoherence_free(5)
    ben_input = BenchmarkInput(
        qc, s.backend(), {"backend_shots": 10, "simulator_shots": 10}
    )

    app_ben = ApplicationBenchmark(
        s,
        ss,
        ben_input,
        analysis=FidelityAnalysis(normalized_fidelity),
        name="GHZ",
    )
    app_ben.run()


def test_jssp():
    from dwave.samplers import (
        SimulatedAnnealingSampler,
    )

    from open_qbench.analysis import FeasibilityRatioAnalysis
    from open_qbench.apps.optimization.jssp import easy_jssp
    from open_qbench.metrics.feasibilities import JSSPFeasibility

    app_ben = OptimizationBenchmark(
        SimulatedAnnealingSampler(),
        BenchmarkInput(easy_jssp()),
        analysis=FeasibilityRatioAnalysis(JSSPFeasibility),
        name="jssp",
    )
    app_ben.run()
