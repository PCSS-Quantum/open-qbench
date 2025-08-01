from collections import Counter

from qiskit.primitives import StatevectorSampler as Sampler

from open_qbench.apps.ghz import ghz_decoherence_free, ghz_direct
from open_qbench.apps.grover import grover_nq
from open_qbench.apps.qaoa import jssp_7q_24d
from open_qbench.apps.qft import prepare_QFT
from open_qbench.apps.qsvm import trained_qsvm_8q
from open_qbench.apps.toffoli import toffoli_circuit
from open_qbench.apps.vqe import uccsd_3q_56d


def test_qsvm__generation():
    qc, params = trained_qsvm_8q()
    assert qc.num_qubits == 8
    assert len(params) == 8
    Sampler().run([(qc, params)]).result()


def test_toffoli_generation():
    qc = toffoli_circuit(5, "11111")
    res = Sampler().run([(qc)]).result()[0].data.meas.get_bitstrings()
    # assert res[0] == "01111"
    assert all(s == "01111" for s in res)


def test_ghz_generation():
    ghz_direct(3)
    ghz_decoherence_free(4)


def test_grover_generation():
    qc = grover_nq(4, 10)
    res = Sampler().run([(qc)]).result()[0].data.meas.get_counts()
    c = Counter(res)
    assert c.most_common(1)[0][0] == bin(10)[2:]


def test_qaoa_generation():
    qc, params = jssp_7q_24d()
    Sampler().run([(qc, params)]).result()


def test_vqe_generation():
    qc, params = uccsd_3q_56d()
    Sampler().run([(qc, params)]).result()


def test_qft():
    num = 24
    qc = prepare_QFT(num)
    res = Sampler().run([(qc)]).result()[0].data.meas.get_bitstrings()
    assert all(s == bin(num)[2:] for s in res)
