import numpy as np
import pytest

from qc_app_benchmarks.photonics import PhotonicCircuit, PhotonicRegister


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
    ph_circuit = PhotonicCircuit.from_tbi_params([1, 1, 1, 1], [1, 2, 3], [np.pi/4]*6)
    assert isinstance(ph_circuit, PhotonicCircuit)
