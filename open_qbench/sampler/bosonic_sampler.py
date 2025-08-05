from abc import abstractmethod
from collections.abc import Iterable

from qiskit.primitives import BasePrimitiveJob, BaseSamplerV2

from open_qbench.photonics import PhotonicCircuit

type PubLike = (
    PhotonicCircuit
    | Iterable[PhotonicCircuit]
    | tuple[PhotonicCircuit, Iterable[float]]
)


class BosonicSampler(BaseSamplerV2):
    @abstractmethod
    def run(
        self,
        pubs: Iterable[PubLike],
        *,
        shots: int | None = None,
    ) -> BasePrimitiveJob:
        """This class differs from BaseSamplerV2 only in types of pubs and the returned job object.

        Args:
            pubs (Iterable[Tuple[PhotonicCircuit, float]]): An iterable of pubs containing PhotonicCircuits
            (should be extended to more general pubs)
            shots (int | None, optional): Number of collected samples.

        Returns:
            Job object of BosonicSampler's result

        """
