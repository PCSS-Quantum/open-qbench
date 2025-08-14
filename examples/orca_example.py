from ptseries.tbi import create_tbi

from open_qbench.apps.circuits.max_cut_orca import (
    max_cut_6_edges_new_input,
    max_cut_6_edges_new_input_double_loop,
)
from open_qbench.fidelity_benchmark import (
    FidelityBenchmark,
)
from open_qbench.metrics.fidelities import (
    create_normalized_fidelity,
)
from open_qbench.sampler.bosonic_sampler import BosonicSampler

# Single loop experiment

# number_of_samples
n_samples = 1000
# #number_of_loops
n_loops = 1

# Define samplers

# ideal sampler
ideal_tbi = create_tbi(n_loops=1)
ideal_sampler = BosonicSampler(ideal_tbi, default_samples=n_samples)

# ORCA sampler
# orca_tbi = create_tbi(
#     tbi_type="PT-1",
#     n_loops = 1,
#     ip_address="192.168.34.2"
# )

# simulator sampler (for testing)
orca_tbi = create_tbi(
    n_loops=1,
    bs_loss=0.01,
    bs_noise=0.01,
    input_loss=0.01,
    detector_efficiency=0.99,
)
orca_sampler = BosonicSampler(orca_tbi, default_samples=n_samples)

# input_state and thetas
input_state = max_cut_6_edges_new_input(return_graph=False, return_input_state=True)[
    "input_state1"
]
thetas = max_cut_6_edges_new_input(return_graph=False, return_input_state=False)

print(input_state)
print(thetas)

# Run the fidelity benchmark

fb = FidelityBenchmark(
    orca_sampler,
    ideal_sampler,
    [input_state, thetas],
    "orca_test_n_loops_1",
)
fb.calculate_accuracy = create_normalized_fidelity(input_state)

res = fb.run()
print(f"{res=}")


# Double loop experiment

# number_of_samples
n_samples = 1000
# #number_of_loops
n_loops = 2

# Define samplers

# ideal sampler
ideal_tbi = create_tbi(n_loops=2)
ideal_sampler = BosonicSampler(ideal_tbi, default_samples=n_samples)

# ORCA sampler
# orca_tbi = create_tbi(
#     tbi_type="PT-1",
#     n_loops = 2,
#     ip_address="192.168.34.2"
# )

# simulator sampler (for testing)
orca_tbi = create_tbi(n_loops=2)  # double-loop doesn't work with noise parameters
orca_sampler = BosonicSampler(orca_tbi, default_samples=n_samples)

# input_state and thetas
input_state = max_cut_6_edges_new_input_double_loop(
    return_graph=False, return_input_state=True
)["input_state1"]
thetas = max_cut_6_edges_new_input_double_loop(
    return_graph=False, return_input_state=False
)

print(input_state)
print(thetas)

# Run the fidelity benchmark

fb = FidelityBenchmark(
    orca_sampler,
    ideal_sampler,
    [input_state, thetas],
    "orca_test_n_loops_2",
)
fb.calculate_accuracy = create_normalized_fidelity(input_state)

res = fb.run()
print(f"{res=}")

############
# Some additional code, generating plots

# def merge_dicts(dicts):
#     ret = defaultdict(int)
#     for d in dicts:
#         for k, v in d.items():
#             ret[k] += v
#     return dict(ret)

# ideal_samples = ideal_tbi.sample(
#         input_state=input_state,
#         theta_list=thetas,
#         n_samples=n_samples)
#
# ideal_samples_sorted = dict(sorted(ideal_samples.items(), key=lambda state: state[0]))
# ideal_labels = list(ideal_samples_sorted.keys())
# ideal_labels = [str(i) for i in ideal_labels]
# ideal_values = list(ideal_samples_sorted.values())
#
#
# orca_samples = orca_tbi.sample(
#         input_state=input_state,
#         theta_list=thetas,
#         n_samples=n_samples)
#
# orca_samples_sorted = dict(sorted(orca_samples.items(), key=lambda state: state[0]))
# orca_labels = list(orca_samples_sorted.keys())
# orca_labels = [str(i) for i in orca_labels]
# orca_values = list(orca_samples_sorted.values())
#
# print(ideal_samples)
# print(orca_samples)
#
# with open('open_qbench/results/ideal_samples' + str(n_loops) + '.txt', 'w') as file:
#     file.write(str(ideal_samples))
#
# with open('open_qbench/results/orca_samples' + str(n_loops) + '.txt', 'w') as file:
#     file.write(str(orca_samples))
#
#
# ### Plots (optional)
# list_of_dicts = [ideal_samples, orca_samples]
# merged_dicts = merge_dicts(list_of_dicts)
#
# master_dict = {}
# for key in merged_dicts:
#     samples = []
#     for d in list_of_dicts:
#         if key in d.keys():
#             sample = d[key]
#         else:
#             sample = 0
#         samples.append(sample)
#
#     master_dict[key] = [samples]
#
# samples_sorted = dict(sorted(master_dict.items(), key=lambda state: state[0]))
# labels = list(samples_sorted.keys())
# labels = [str(i) for i in labels]
# values = np.array(list(samples_sorted.values()))[:, 0]
#
# fig, axs = plt.subplots(2, 2)
# fig.set_size_inches(8, 8)
#
# axs[0, 0].bar(labels, values[:, 0], tick_label=labels, alpha = 1)
# plt.setp(axs[0, 0].get_xticklabels(), rotation=45, ha='right')
#
# axs[0, 1].bar(labels, values[:, 1], tick_label=labels, alpha = 1)
# plt.setp(axs[0, 1].get_xticklabels(), rotation=45, ha='right')
#
# axs[1, 0].bar(labels, values[:, 0] - values[:, 1], tick_label=labels, alpha = 1)
# plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha='right')
#
# axs[1, 1].bar(labels, values[:, 0], tick_label=labels, alpha = 0.5)
# axs[1, 1].bar(labels, values[:, 1], tick_label=labels, alpha = 0.5)
# plt.setp(axs[1, 1].get_xticklabels(), rotation=45, ha='right')
#
# plt.tight_layout()
# plt.savefig("open_qbench/results/orca_benchmark_plots.pdf")
