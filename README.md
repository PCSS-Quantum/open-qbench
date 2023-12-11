# Application Performance Benchmarks for Quantum Computers
This repository contains a series of quantum application benchmarks meant to be run on different physical quantum devices to measure their performance and fidelity of the results [1].

We identify a set of quantum algorithms, which exemplify the common approaches to performing quantum computations in different application areas. These include both the currently most commonly used near-term variational algorithms and routines used in the fault-tolerant QC.

In order to evaluate a given quantum system’s ability to execute an algorithm, we choose a single quantum circuit, which by design is meant to represent the most typical single execution, either hybrid or purely quantum. This is especially worth noting in the case of variational algorithms, where we do not intend to perform a full run, optimizing the parameters, but rather fix the parameters in place and estimate the fidelity on a single non-parameterized circuit. This is done carefully, in order to avoid cases where the ideal distribution is close to uniform.

The logical quantum circuits for each benchmark are compiled into OpenQASM 2.0/3.0 assuming all-to-all connectivity and {Rx, Ry, Rz, CNOT} as the basis gate set. For execution on real quantum backends, these circuits can be freely recompiled and optimized, as long as they remain logically equivalent to the ones delivered within the described suite. This also means that while error mitigation is not meant to be a part of these benchmarks, error suppression techniques like dynamic decoupling can be used.

The following main metrics based on best practices discussed in [2] have been identified:
- Execution Time: time spent on quantum simulator or hardware backend running the circuit;
- Circuit Depth: depth of the circuit after transpiling it to the basis gates set defined as {Rx, Ry, Rz, CNOT}
- Fidelity: a measure of how well the simulator or hardware runs a particular benchmark;

The following benchmarks are currently implemented:

| Benchmark   | Number of Qubits |
|-------------|------------------|
| VQE (UCCSD) | 3                |
| QAOA (JSSP) | 7                |
| QSVM        | 8                |
| QFT         | arbitrary        |
| Grover      | arbitrary        |
| Toffoli     | 5                |

## Installation
To install this package with minimal requirements use `pip`:
```
pip install .
```
The VQE and QSVM benchmarks require some additional dependencies, so in order to run these, install with:
```
pip install .[VQE,QSVM]
```

## References
[1]: [Application Performance Benchmarks for Quantum Computers](https://arxiv.org/abs/2310.13637)
[2]: [Application-Oriented Performance Benchmarks for Quantum Computing](https://arxiv.org/abs/2110.03137)
