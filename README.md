<p align="center">
  <img src="./docs/_static/logo-dark.svg" />
</p>


# Open QBench
[![PyPI - Version](https://img.shields.io/pypi/v/open-qbench.svg?color=blue)](https://pypi.org/project/open-qbench/)
[![License](https://img.shields.io/github/license/PCSS-quantum/open-qbench)](https://github.com/PCSS-Quantum/open-qbench/blob/qbench_v2/LICENSE)


Open QBench is an open-source software framework for defining and executing benchmarks across the quantum-classical stack. It offers support for benchmarks on the:

- hybrid level (quantum-classical workflows)
- high level (quantum algorithms)
- low level (compiled quantum programs)

The framework is supports gate-based quantum computers, photonic systems (boson samplers), and quantum annealers.

Beyond the framework, this package also includes a suite of pre-implemented, high-level quantum application benchmarks. These are specifically crafted to evaluate the performance and fidelity of results on diverse physical quantum devices.

## Installation
### Using uv (Recommended)
First, create and activate a virtual environment:
```
uv venv
```
To install the core dependencies of the package, run:
```
uv pip install open-qbench
```
### Optional dependencies (Extras)
We provide a number of optional dependencies (referred to as "extras") for executing specific benchmarks or for enabling support for various quantum hardware providers.

Available extras include:

- Benchmarks: `VQE`, `QSVM`
- Providers: `IBM`, `AQT`

To install specific optional dependencies, specify them in your command. You can specify multiple optionals in a single command. For example, to run the VQE benchmark on an IBM Quantum machine, you would run:
```
uv pip install "open-qbench[VQE, IBM]"
```
You can combine any of the available extras as needed.

> [!NOTE]
> To install the PT-Series SDK used for running experiments on the ORCA Computing systems, you will need to get access to it through the website at https://sdk.orcacomputing.com/ and install it separately.

### Using pip
The package can also be installed with pip.
To install the core dependencies run:
```
pip install open-qbench
```
To install with specific optional dependencies (e.g., `VQE` and `IBM`):
```
pip install "open-qbench[VQE,IBM]"
```

## Usage
### How to run a benchmark
This example shows how to execute a simple application benchmark using a Grover circuit on simulated IBM Quantum hardware (check [Installation](#installation) to see how to enable IBM support).

First define samplers used for collecting distributions.

```python
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import Sampler
from qiskit_ibm_runtime.fake_provider import FakeGeneva

ideal_sampler = AerSampler(default_shots=1000)
backend_sampler = Sampler(FakeGeneva())
```

Then use Open QBench to generate a quantum circuit and create your benchmark by defining input and a function used to calculate fidelity.

```python
from open_qbench import ApplicationBenchmark
from open_qbench.apps.circuits import grover
from open_qbench.core import BenchmarkInput
from open_qbench.fidelities import normalized_fidelity

qc = grover.grover_nq(3, 6)
backend = backend_sampler.backend()
benchmark_input = BenchmarkInput(qc, backend)

ab = ApplicationBenchmark(
    backend_sampler,
    ideal_sampler,
    benchmark_input,
    name="Grover_benchmark",
    accuracy_measure=normalized_fidelity,
)

ab.run()
print(ab.result)

```

## Contributing
We welcome contributions from the community! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for a detailed guide.

## License
[Apache License 2.0](https://github.com/PCSS-Quantum/open-qbench/blob/qbench_v2/LICENSE)
