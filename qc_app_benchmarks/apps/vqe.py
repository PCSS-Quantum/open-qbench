from typing import Tuple

from qiskit import QuantumCircuit

from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.settings import settings

settings.use_pauli_sum_op = False


def uccsd_3q_56d() -> Tuple[QuantumCircuit, tuple]:
    """Returns a 3-qubit UCCSD circuit with normalized depth 56, as in the document.
    Returned parameters are chosen arbitrarily, so that the final distribution is
    not uniform."""

    mol = MoleculeInfo(["Li", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 1.55)])
    driver = PySCFDriver.from_molecule(mol, basis="sto-3g")

    problem = driver.run()
    transformer = ActiveSpaceTransformer((1, 1), 3)
    problem = transformer.transform(problem)

    n_particles = problem.num_particles
    n_orbitals = problem.num_spatial_orbitals

    p_mapper = ParityMapper(n_particles)
    mapper = problem.get_tapered_mapper(p_mapper)

    init_state = HartreeFock(n_orbitals, n_particles, mapper)
    ansatz = UCCSD(
        qubit_mapper=mapper,
        num_particles=n_particles,
        num_spatial_orbitals=n_orbitals,
        initial_state=init_state,
        generalized=False,
        reps=1,
    )
    ansatz.measure_all()
    ansatz.name = "UCCSD_LiH_3q"
    params = [0.08670186, 0.41080424, 0.96417694, 0.17362798]

    return ansatz, params
