"""Running this example requires adding your SSH key to https://sdk.orcacomputing.com/ and installing with pip install .[ORCA]"""

import numpy as np
from qc_app_benchmarks.photonics import PhotonicCircuit, PhotonicRegister
from qc_app_benchmarks.sampler.orca_sampler import OrcaSampler
from qc_app_benchmarks.fidelities import normalized_fidelity
from qc_app_benchmarks.fidelity_benchmark import BenchmarkSuite

ph_circuit1 = PhotonicCircuit(input_state=[1, 1, 1, 1])
ph_circuit1.bs(np.pi/4, 0, 1)
ph_circuit1.bs(np.pi/4, 1, 2)
ph_circuit1.bs(np.pi/4, 2, 3)
ph_circuit1.bs(np.pi/4, 0, 2)
ph_circuit1.bs(np.pi/4, 1, 3)
ph_circuit1.bs(np.pi/4, 0, 3)

ph_circuit2 = PhotonicCircuit(input_state=[1, 1, 1, 0])
ph_circuit2.bs(np.pi/4, 0, 1)
ph_circuit2.bs(np.pi/4, 0, 3)
ph_circuit2.bs(np.pi/4, 2, 3)
ph_circuit2.bs(np.pi/4, 1, 2)
ph_circuit2.bs(np.pi/4, 0, 2)
ph_circuit2.bs(np.pi/4, 1, 3)

ideal_sampler = OrcaSampler(default_shots=1024)
backend_sampler = OrcaSampler(default_shots=1024)

suite = BenchmarkSuite(
    backend_sampler=backend_sampler,
    ideal_sampler=ideal_sampler,
    calculate_accuracy=normalized_fidelity,
    name="test_suite",
)
suite.add_benchmarks(
    [
        [(ph_circuit1, [np.pi/4]*6),],
        [(ph_circuit2, [np.pi/4]*6),]
    ]
)
suite.run_all()
print("Results:")
for res in suite.results:
    print(
        f"{res.name:>15}: depth = {res.input_properties['normalized_depth']}, fidelity = {res.average_fidelity}"
    )
suite.save_results("test_res")
