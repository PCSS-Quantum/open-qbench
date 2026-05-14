from collections import Counter
from typing import Any

import dimod
from qiskit import QuantumCircuit
from qiskit.primitives import BaseSamplerV1, BaseSamplerV2, BitArray  # , SamplerPubLike

try:
    from qlauncher import QLauncher
    from qlauncher.base import Algorithm, Backend, Problem
    from qlauncher.base.adapter_structure import get_formatter
except ImportError:

    class Problem:  # type: ignore[no-redef]
        pass

    class Algorithm:
        pass

    class Backend:
        pass


from open_qbench.photonics import PhotonicCircuit


class BenchmarkSampler:
    def __init__(
        self,
        sampler: BaseSamplerV1
        | BaseSamplerV2
        | dimod.Sampler
        | tuple[Algorithm, Backend],
        shots=1024,
        **sampling_kwargs,
    ) -> None:
        self.sampler = sampler
        self.shots = shots
        self.kwargs = sampling_kwargs

    def get_counts(
        self, sampler_input: QuantumCircuit | PhotonicCircuit | Problem
    ) -> dict[Any, int]:
        """Get sample counts after running sampler on input."""
        try:
            max_shots = self.sampler._backend.configuration().max_shots  # works for AQT
        except AttributeError:
            max_shots = 1000000  # assume large number and let fail in run()

        merged_counts = Counter()
        shots_to_execute = self.shots
        while shots_to_execute:
            if shots_to_execute > max_shots:
                shots = max_shots
                shots_to_execute -= max_shots
            else:
                shots = shots_to_execute
                shots_to_execute -= shots
            if isinstance(sampler_input, PhotonicCircuit) and isinstance(
                self.sampler, BaseSamplerV2
            ):
                counts = self.sampler.run(
                    [sampler_input], shots=shots, **self.kwargs
                ).result()[0]
            elif isinstance(sampler_input, QuantumCircuit):
                if isinstance(self.sampler, BaseSamplerV2):
                    sampler_data = (
                        self.sampler.run([sampler_input], shots=shots, **self.kwargs)
                        .result()[0]
                        .data
                    )
                    counts = BitArray.concatenate_bits(
                        list(sampler_data.values())
                    ).get_counts()
                elif isinstance(self.sampler, BaseSamplerV1):
                    result = self.sampler.run(
                        circuits=[sampler_input], shots=shots, **self.kwargs
                    ).result()
                    counts = result.quasi_dists[0].binary_probabilities()
                    counts = {
                        k: int(v * result.metadata[0]["shots"])
                        for (k, v) in counts.items()
                    }
                else:
                    raise NotImplementedError
            elif isinstance(sampler_input, Problem) and isinstance(
                self.sampler, dimod.Sampler
            ):
                bqm = get_formatter(sampler_input.__class__, "bqm")(sampler_input)
                result = self.sampler.sample(bqm, num_reads=shots, **self.kwargs)
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

            merged_counts += Counter(dict(counts))

        return dict(merged_counts)
