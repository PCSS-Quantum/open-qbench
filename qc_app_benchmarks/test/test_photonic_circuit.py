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
