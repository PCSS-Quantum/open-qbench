import pytest
import numpy as np
from qc_app_benchmarks.photonics import PhotonicCircuit, PhotonicRegister
from qc_app_benchmarks.sampler.orca_sampler import OrcaSampler


def test_conversion1():
    ph_circuit = PhotonicCircuit(PhotonicRegister(4))
    ph_circuit.input_state = [1, 1, 1, 1]
    ph_circuit.bs(np.pi/4, 0, 1)
    ph_circuit.bs(np.pi/4, 1, 2)
    ph_circuit.bs(np.pi/4, 0, 2)
    ph_circuit.bs(np.pi/4, 1, 3)
    ph_circuit.bs(np.pi/4, 0, 3)
    _, new_thetas, loop_lengths = OrcaSampler()._extract_lengths((ph_circuit, [np.pi/4]*5))
    assert loop_lengths == [1, 2, 3]
    assert new_thetas == [np.pi/4, np.pi/4, 0, np.pi/4, np.pi/4, np.pi/4]


def test_conversion2():
    ph_circuit = PhotonicCircuit(PhotonicRegister(4))
    ph_circuit.input_state = [1, 1, 1, 1]
    ph_circuit.bs(np.pi/4, 0, 1)
    ph_circuit.bs(np.pi/4, 0, 3)
    ph_circuit.bs(np.pi/4, 2, 3)
    ph_circuit.bs(np.pi/4, 1, 2)
    ph_circuit.bs(np.pi/4, 0, 2)
    ph_circuit.bs(np.pi/4, 1, 3)
    _, new_thetas, loop_lengths = OrcaSampler()._extract_lengths((ph_circuit, [np.pi/4]*6))
    assert loop_lengths == [1, 3, 1, 1, 2]
    assert new_thetas == [np.pi/4, 0, 0, np.pi/4, 0, 0, np.pi/4, 0, np.pi/4, 0, np.pi/4, np.pi/4]


def test_validation():
    ph_circuit = PhotonicCircuit(PhotonicRegister(4))
    ph_circuit.input_state = [1, 1, 1, 1]
    ph_circuit.bs(np.pi/4, 0, 1)
    ph_circuit.bs(np.pi/4, 1, 2)
    ph_circuit.bs(np.pi/4, 2, 3)
    ph_circuit.bs(np.pi/4, 0, 2)
    ph_circuit.bs(np.pi/4, 1, 3)
    ph_circuit.bs(np.pi/4, 0, 3)

    with pytest.raises(TypeError):
        _ = OrcaSampler().run([(ph_circuit, [np.pi/4]*5+[np.pi]), (ph_circuit, [np.pi/4]*6)])


def test_sampler():
    # Valid circuit 1
    ph_circuit1 = PhotonicCircuit(PhotonicRegister(4))
    ph_circuit1.input_state = [1, 1, 1, 1]
    ph_circuit1.bs(np.pi/4, 0, 1)
    ph_circuit1.bs(np.pi/4, 1, 2)
    ph_circuit1.bs(np.pi/4, 2, 3)
    ph_circuit1.bs(np.pi/4, 0, 2)
    ph_circuit1.bs(np.pi/4, 1, 3)
    ph_circuit1.bs(np.pi/4, 0, 3)

    # Valid circuit 2
    ph_circuit2 = PhotonicCircuit(PhotonicRegister(4))
    ph_circuit2.input_state = [1, 1, 1, 0]
    ph_circuit2.bs(np.pi/4, 0, 1)
    ph_circuit2.bs(np.pi/4, 1, 2)
    ph_circuit2.bs(np.pi/4, 2, 3)
    ph_circuit2.bs(np.pi/4, 0, 1)
    ph_circuit2.bs(np.pi/4, 1, 2)
    ph_circuit2.bs(np.pi/4, 2, 3)

    job = OrcaSampler().run([(ph_circuit1, [np.pi/4]*6), (ph_circuit2, [np.pi/4]*6)], shots=1000)

    assert isinstance(job.result()[0], dict)

    assert isinstance(job.result()[1], dict)
