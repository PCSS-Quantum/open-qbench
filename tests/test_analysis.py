from open_qbench.analysis import FeasibilityRatioAnalysis, FidelityAnalysis
from open_qbench.apps.optimization.jssp import easy_jssp
from open_qbench.core.benchmark import BenchmarkInput, BenchmarkResult
from open_qbench.metrics.feasibilities import JSSPFeasibility
from open_qbench.metrics.fidelities import normalized_fidelity


def test_fidelity_analysis():
    dist_a = {
        "1101": 127,
        "0010": 108,
        "0101": 131,
        "0000": 98,
        "0100": 65,
        "0001": 90,
        "1010": 121,
        "1110": 115,
        "0111": 14,
        "1100": 30,
        "1111": 62,
        "1000": 41,
        "1011": 5,
        "1001": 2,
        "0110": 11,
        "0011": 4,
    }

    dist_b = {
        "1101": 0.1240234375,
        "0010": 0.10546875,
        "0101": 0.1279296875,
        "0000": 0.095703125,
        "0100": 0.0634765625,
        "0001": 0.087890625,
        "1010": 0.1181640625,
        "1110": 0.1123046875,
        "0111": 0.013671875,
        "1100": 0.029296875,
        "1111": 0.060546875,
        "1000": 0.0400390625,
        "1011": 0.0048828125,
        "1001": 0.001953125,
        "0110": 0.0107421875,
        "0011": 0.00390625,
    }

    res = BenchmarkResult("test", None, {}, {})
    res.execution_data["counts_ideal"] = dist_a
    res.execution_data["counts_backend"] = dist_b
    analysis = FidelityAnalysis(normalized_fidelity)
    res = analysis.run(res)
    assert res.metrics["fidelity"] == 1.0


def test_feasibility_ratio_analysis():
    dist_a = {"1111": 99, "1000": 1}
    res = BenchmarkResult("test", BenchmarkInput(program=easy_jssp()), {}, {})
    res.execution_data["counts_backend"] = dist_a

    analysis = FeasibilityRatioAnalysis(JSSPFeasibility)

    res = analysis.run(res)

    assert res.metrics["feasibility_ratio"] == 0.99
