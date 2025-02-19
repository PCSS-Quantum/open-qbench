from abc import ABC, abstractmethod


class SamplerResult:
    pass


class BaseBenchmarkSampler(ABC):
    @abstractmethod
    def run(self, sampler_input, num_samples: int | None = None) -> SamplerResult:
        """This method is a generalization of BaseSamplerV2.run() from Qiskit.
        The idea is to allow not only pubs (list of circuits or tuples (circuit, parameter_values)),
        but also a list of tuples of parameters for bosonic samplers.

        Args:
            sampler_input (_type_): A PUB or a list of tuples of circuit parameters
            num_samples (int | None, optional): Number of shots to sample for each input
            that does not specify its own. Defaults to None.

        Returns:
            SamplerResult: The result of sampling the input.
        """
        pass
