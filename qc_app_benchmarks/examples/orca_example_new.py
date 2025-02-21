"""Running this example requires adding your SSH key to https://sdk.orcacomputing.com/ and installing with pip install .[ORCA]"""

from ptseries.tbi import PT

from qc_app_benchmarks.sampler import BosonicSampler


class OrcaSampler(BosonicSampler):
    """This class is separate from the library as the ptseries SDK
    is not public and we want to avoid adding it as dependency."""

    def run(self, pubs, *, shots=None):
        # TODO: implement run using ptseries API
        pass
