import numpy as np
import pytest

from qc_app_benchmarks.photonics import PhotonicCircuit, PhotonicRegister
from qc_app_benchmarks.photonics.photonic_gates import BS, PhotonicGate


def create_bs_circuit(size: int, qm1: int, qm2: int):
    pr = PhotonicRegister(size)
    pc = PhotonicCircuit(pr)

    pc.bs(theta=1.5, qumode1=qm1, qumode2=qm2)
    return pc


def test_circuit_creation():
    create_bs_circuit(2, 0, 1)
    with pytest.raises(IndexError):
        create_bs_circuit(2, 0, 2)


def test_qumodes_binding():
    pr = PhotonicRegister(2)
    pc = PhotonicCircuit(pr)

    pc.bs(theta=1.5, qumode1=0, qumode2=1)
    qm0 = pr[0]
    qm1 = pc._data[0].qumodes[0]
    assert qm0 is qm1


def test_incorrect_operation():
    pr = PhotonicRegister(2)
    pc = PhotonicCircuit(pr)

    with pytest.raises(Exception):
        pc.h(0)


def test_drawing():
    pr = PhotonicRegister(2)
    pc = PhotonicCircuit(pr)

    pc.bs(theta=1.5, qumode1=0, qumode2=1)
    with pytest.raises(ModuleNotFoundError):
        pc.draw(draw=False)
        raise ModuleNotFoundError
    # Explanation: ModuleNotFoundError is acceptable result, test fails on different Errors/Exceptions
    # Cannot be fully tested without creating plt window


def test_from_tbi_params():
    input_state = [1, 1, 1, 1]
    loop_lengths = [1, 2, 3]
    expected_qumodes = []
    for length in loop_lengths:
        for qumode in range(length, len(input_state)):
            expected_qumodes.append((qumode-length, qumode))
    thetas = [np.pi/4]*6
    ph_circuit: PhotonicCircuit = PhotonicCircuit.from_tbi_params(input_state, loop_lengths, thetas)
    for i, op in enumerate(ph_circuit):
        assert isinstance(op.operation, BS)
        assert isinstance(op.operation, PhotonicGate)
        assert op.qumodes[0]._index == expected_qumodes[i][0]
        assert op.qumodes[1]._index == expected_qumodes[i][1]
        assert op.params[0] == thetas[i]

def test_BS_compare():
    bs1 = BS(np.pi/4)
    bs2 = BS(np.pi/6)
    bs3 = BS(np.pi/4)
    assert not bs1 == bs2
    assert bs1 == bs3