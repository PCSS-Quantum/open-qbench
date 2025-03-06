from qc_app_benchmarks.photonics import PhotonicCircuit, photonic_circuit
from examples.orca_sampler import OrcaSampler


from ptseries.tbi import create_tbi
from qc_app_benchmarks.apps.max_cut_orca import max_cut_thetas_6_edges, max_cut_6_edges_new_input
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


def merge_dicts(dicts):
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)


# Define samplers

# ideal sampler


# TODO - replace it with a newer version

# ideal_sampler = BosonicSampler(ideal_tbi)
# ideal sampler definition
# ideal_tbi = create_tbi(
#     n_loops = 1
# )

# ORCA sampler

# TODO - replace it with a newer version

# orca_tbi = create_tbi(
#     tbi_type="PT-1",
#     n_loops = 1,
#     ip_address="169.254.109.10"
# )

# orca_sampler = BosonicSampler(orca_tbi, default_samples=10)


# simulator sampler
# # TODO - just a mock-up
# orca_tbi = create_tbi(
#     n_loops = 1,
#     bs_loss=0.01,
#     bs_noise=0.01,
#     input_loss=0.01,
#     detector_efficiency=0.99
# )


# Experiments
# number_of_samples
n_samples = 200
# number_of_loops
n_loops = 1

ideal_sampler = OrcaSampler(default_shots=n_samples)
#backend_sampler = OrcaSampler(default_options={"tbi_type": "PT-1", "url": "169.254.109.10"}, default_shots=n_samples)
backend_sampler = OrcaSampler(default_shots=n_samples)

input_state = max_cut_6_edges_new_input(return_graph=False, return_input_state=True)['input_state1']
thetas = max_cut_6_edges_new_input(return_graph=False, return_input_state=False)

print(input_state)
print(thetas)


photonic_circuit = PhotonicCircuit().from_tbi_params(input_state, [1], thetas)

ideal_job = ideal_sampler.run([(photonic_circuit, thetas),])
orca_job = backend_sampler.run([(photonic_circuit, thetas),], options={
                               "n_loops": 1, "bs_loss": 0.01, "bs_noise": 0.01, "input_loss": 0.01, "detector_efficiency": 0.99})

ideal_samples = ideal_job.result()[0]
orca_samples = orca_job.result()[0]

print(ideal_samples)
print(orca_samples)

# ideal_samples = ideal_tbi.sample(
#     input_state=input_state,
#     theta_list=thetas,
#     n_samples=n_samples)

ideal_samples_sorted = dict(sorted(ideal_samples.items(), key=lambda state: state[0]))
ideal_labels = list(ideal_samples_sorted.keys())
ideal_labels = [str(i) for i in ideal_labels]
ideal_values = list(ideal_samples_sorted.values())


# orca_samples = orca_tbi.sample(
#     input_state=input_state,
#     theta_list=thetas,
#     n_samples=n_samples)

orca_samples_sorted = dict(sorted(orca_samples.items(), key=lambda state: state[0]))
orca_labels = list(orca_samples_sorted.keys())
orca_labels = [str(i) for i in orca_labels]
orca_values = list(orca_samples_sorted.values())

print(ideal_samples)
print(orca_samples)

import os
os.makedirs('qc_app_benchmarks/results/ideal_samples', exist_ok=True)
os.makedirs('qc_app_benchmarks/results/orca_samples', exist_ok=True)

with open('qc_app_benchmarks/results/ideal_samples' + str(n_loops) + '.txt', 'w') as file:
    file.write(str(ideal_samples))

with open('qc_app_benchmarks/results/orca_samples' + str(n_loops) + '.txt', 'w') as file:
    file.write(str(orca_samples))


# Plots (optional)
list_of_dicts = [ideal_samples, orca_samples]
merged_dicts = merge_dicts(list_of_dicts)

master_dict = {}
for key in merged_dicts:
    samples = []
    for d in list_of_dicts:
        if key in d.keys():
            sample = d[key]
        else:
            sample = 0
        samples.append(sample)

    master_dict[key] = [samples]

samples_sorted = dict(sorted(master_dict.items(), key=lambda state: state[0]))
labels = list(samples_sorted.keys())
labels = [str(i) for i in labels]
values = np.array(list(samples_sorted.values()))[:, 0]

fig, axs = plt.subplots(2, 2)
fig.set_size_inches(8, 8)

axs[0, 0].bar(labels, values[:, 0], tick_label=labels, alpha=1)
plt.setp(axs[0, 0].get_xticklabels(), rotation=45, ha='right')

axs[0, 1].bar(labels, values[:, 1], tick_label=labels, alpha=1)
plt.setp(axs[0, 1].get_xticklabels(), rotation=45, ha='right')

axs[1, 0].bar(labels, values[:, 0] - values[:, 1], tick_label=labels, alpha=1)
plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha='right')

axs[1, 1].bar(labels, values[:, 0], tick_label=labels, alpha=0.5)
axs[1, 1].bar(labels, values[:, 1], tick_label=labels, alpha=0.5)
plt.setp(axs[1, 1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig("qc_app_benchmarks/results/orca_benchmark_plots.pdf")
