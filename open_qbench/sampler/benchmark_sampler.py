from typing import Any

import dimod
from qiskit import QuantumCircuit
from qiskit.primitives import BackendSamplerV2, BaseSamplerV2
from qiskit.providers import BackendV1, BackendV2
from qiskit_ibm_runtime import Sampler, SamplerV2
from qlauncher import QLauncher
from qlauncher.base import Algorithm, Backend, Problem
from qlauncher.base.adapter_structure import get_formatter

from open_qbench.photonics import PhotonicCircuit


class BenchmarkSampler:
    def __init__(
        self,
        sampler: BaseSamplerV2 | dimod.Sampler | tuple[Algorithm, Backend],
        shots: int = 1024,
        backend_name: str | None = None,
        **sampling_kwargs,
    ) -> None:
        self.sampler = sampler
        self.shots = shots
        self._backend_name = backend_name
        self.kwargs = sampling_kwargs

    @property
    def backend_name(self) -> str | None:
        if self._backend_name is not None:
            return self._backend_name

        if isinstance(self.sampler, BackendSamplerV2 | Sampler | SamplerV2):
            print(type(self.sampler.backend))
            if isinstance(self.sampler.backend, BackendV1):
                return self.sampler.backend.name()
            elif isinstance(self.sampler.backend, BackendV2):
                return self.sampler.backend.name
            elif callable(self.sampler.backend):
                return self.sampler.backend().name
            else:
                return None

        return str(self.sampler).rsplit(".", maxsplit=1)[-1].split("'")[0]

    def get_counts(
        self, sampler_input: QuantumCircuit | PhotonicCircuit | Problem
    ) -> dict[Any, int]:
        """Get sample counts after running sampler on input"""
        if isinstance(sampler_input, PhotonicCircuit) and isinstance(
            self.sampler, BaseSamplerV2
        ):
            counts = self.sampler.run(
                [sampler_input], shots=self.shots, **self.kwargs
            ).result()[0]
        elif isinstance(sampler_input, QuantumCircuit) and isinstance(
            self.sampler, BaseSamplerV2
        ):
            sampler_results = self.sampler.run(
                [sampler_input], shots=self.shots, **self.kwargs
            ).result()[0]
            counts = sampler_results.join_data().get_counts()

        elif isinstance(sampler_input, Problem) and isinstance(
            self.sampler, dimod.Sampler
        ):
            bqm = get_formatter(sampler_input.__class__, "bqm")(sampler_input)
            result = self.sampler.sample(bqm, num_reads=self.shots, **self.kwargs)
            counts = {}
            for value, occ in zip(
                result.record.sample, result.record.num_occurrences, strict=True
            ):
                bitstring = "".join(map(str, value))
                counts[bitstring] = counts.get(bitstring, 0) + occ
        elif isinstance(sampler_input, Problem) and isinstance(self.sampler, tuple):
            alg, backend = self.sampler
            launcher = QLauncher(sampler_input, alg, backend)
            res = launcher.run()
            counts = {
                k: int(round(v * res.num_of_samples, 0))
                for k, v in res.distribution.items()
            }
        else:
            raise NotImplementedError

        return counts
