import numpy as np
import pytest
from qiskit.circuit import CircuitError

from open_qbench.photonics import PhotonicCircuit, PhotonicRegister
from open_qbench.photonics.photonic_gates import BS, PhotonicGate, Qumode


def _lists_match(l1: list, l2: list, exact_is: bool = False) -> bool:
    def comp(e1, e2):
        if exact_is:
            return e1 is e2
        return e1 == e2

    if len(l1) != len(l2):
        return False
    return all(comp(e1, e2) for e1, e2 in zip(l1, l2, strict=False))


def create_bs_circuit(size: int, qm1: int, qm2: int):
    pr = PhotonicRegister(size)
    pc = PhotonicCircuit(pr)

    pc.bs(theta=1.5, qumode1=qm1, qumode2=qm2)


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


def test_qumodes_assignment():
    pr1 = PhotonicRegister(2)
    pr2 = PhotonicRegister(2)
    pc = PhotonicCircuit(pr1, pr2)

    qms: list[Qumode] = [pr1[0], pr2[0]]

    pc.bs(theta=1.5, qumode1=0, qumode2=2)
    pc.bs(theta=1.5, qumode1=[0], qumode2=[2])
    pc.bs(theta=1.5, qumode1=slice(0, 1), qumode2=slice(2, 3))
    pc.bs(theta=1.5, qumode1=qms[0], qumode2=qms[1])
    pc.bs(theta=1.5, qumode1=[qms[0]], qumode2=[qms[1]])

    for inst in pc._data:
        assert _lists_match(list(inst.qumodes), qms, exact_is=True)


def test_qumode_broadcast_expand_right():
    """Test if broadcasting correctly expands a longer right side, also test whole register assignment"""
    pr1 = PhotonicRegister(2)
    pr2 = PhotonicRegister(2)
    pc = PhotonicCircuit(pr1, pr2)

    pc.bs(theta=1.5, qumode1=0, qumode2=pr2)

    assert _lists_match(list(pc._data[0].qumodes), [pr1[0], pr2[0]], exact_is=True)
    assert _lists_match(list(pc._data[1].qumodes), [pr1[0], pr2[1]], exact_is=True)


def test_qumode_broadcast_expand_left():
    """Test if broadcasting correctly expands a longer left side, also test whole register assignment"""
    pr1 = PhotonicRegister(2)
    pr2 = PhotonicRegister(2)
    pc = PhotonicCircuit(pr1, pr2)

    pc.bs(theta=1.5, qumode1=pr1, qumode2=2)

    assert _lists_match(list(pc._data[0].qumodes), [pr1[0], pr2[0]], exact_is=True)
    assert _lists_match(list(pc._data[1].qumodes), [pr1[1], pr2[0]], exact_is=True)


def test_incorrect_operation():
    pr = PhotonicRegister(2)
    pc = PhotonicCircuit(pr)

    with pytest.raises(CircuitError):
        pc.h(0)

    with pytest.raises(CircuitError):
        pc.x(0)

    with pytest.raises(CircuitError):
        pc.cx(0, 1)

    with pytest.raises(CircuitError):
        pc.z(0)

    with pytest.raises(CircuitError):
        pc.rx(np.pi, 0)

    pc.bs(np.pi, 0, 1)


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
            expected_qumodes.append((qumode - length, qumode))
    thetas = [np.pi / 4] * 6
    ph_circuit: PhotonicCircuit = PhotonicCircuit.from_tbi_params(
        input_state, loop_lengths, thetas
    )
    for i, op in enumerate(ph_circuit):
        assert isinstance(op.operation, BS)
        assert isinstance(op.operation, PhotonicGate)
        assert op.qumodes[0]._index == expected_qumodes[i][0]
        assert op.qumodes[1]._index == expected_qumodes[i][1]
        assert op.params[0] == thetas[i]


def test_BS_compare():
    bs1 = BS(np.pi / 4)
    bs2 = BS(np.pi / 6)
    bs3 = BS(np.pi / 4)
    assert bs1 != bs2
    assert bs1 == bs3
