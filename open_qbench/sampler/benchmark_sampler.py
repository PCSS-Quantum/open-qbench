from typing import Any

import dimod
from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV2  # , SamplerPubLike
from qlauncher.base import Algorithm, Backend, Problem
from qlauncher.base.adapter_structure import get_formatter

from open_qbench.photonics import PhotonicCircuit


class BenchmarkSampler:
    def __init__(
        self,
        sampler: BaseSamplerV2 | dimod.Sampler | tuple[Algorithm, Backend],
        shots=1024,
        **sampling_kwargs,
    ) -> None:
        self.sampler = sampler
        self.shots = shots
        self.kwargs = sampling_kwargs

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
            counts = (
                self.sampler.run([sampler_input], shots=self.shots, **self.kwargs)
                .result()[0]
                .data.meas.get_counts()
            )
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
        else:
            raise NotImplementedError

        return counts
