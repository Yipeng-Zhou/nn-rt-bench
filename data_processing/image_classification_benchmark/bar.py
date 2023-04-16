from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# extract the data from the file "extra_benchmarks.csv"
extra_data = pd.read_csv("extra_benchmarks_fp32.csv")
models = extra_data["model"].values
size = extra_data["size(MB)"].values
short_name = extra_data["short_name"].values

positions = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20, 21, 22, 23, 24]

# extract the data from the file "extra_benchmarks_quantization.csv"
extra_data = pd.read_csv("extra_benchmarks_uint8.csv")
models_q = extra_data["model"].values
size_q = extra_data["size(MB)"].values
short_name_q = extra_data["short_name"].values

positions_q = [3, 5, 7, 9, 11, 13, 15, 17]

# merge short names
short_names = list(range(25))
i = j = 0
for position in positions:
    short_names[position] = short_name[i]
    i = i + 1
for position in positions_q:
    short_names[position] = short_name_q[j]
    j = j + 1

# extract the data from the folder "benchmarks"
data_folder = "benchmarks/"

## Non-quantization
job_elapsed = []
instructions_retired = []
l1_misses = []
l1_miss_ratio = []
l2_misses = []
l2_miss_ratio = []
cpu_clock_count = []

for model in models:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed += [np.mean(data["job_elapsed(seconds)"].values)]
    instructions_retired += [np.mean(data["instructions_retired"].values)]
    l1_misses += [np.mean(data["job_l1_misses"].values)]
    l1_miss_ratio += [np.mean(data["job_l1_miss_ratio(%%)"].values)]
    l2_misses += [np.mean(data["job_l2_misses"].values)]
    l2_miss_ratio += [np.mean(data["job_l2_miss_ratio(%%)"].values)]
    cpu_clock_count += [np.mean(data["cpu_clock_count"].values)]

bandwidth = np.divide(l2_misses, job_elapsed)
instructions_per_second = np.divide(instructions_retired, job_elapsed)
instructions_per_clock = np.divide(instructions_retired, cpu_clock_count)

flag_sum = []

for model in models:
    data = pd.read_csv(data_folder+model+"/"+"prediction_output.csv")
    flag_sum += [np.sum(data["prediction_flag"].values)]

examples = np.zeros(np.array(flag_sum).shape) + 10
top_1_accuracy = np.divide(flag_sum, examples)

# print(size)
# print(job_elapsed)
# print(top_1_accuracy)
# print(l2_misses)
# print(bandwidth)

## Quantization
job_elapsed_q = []
instructions_retired_q = []
l1_misses_q = []
l1_miss_ratio_q = []
l2_misses_q = []
l2_miss_ratio_q = []
cpu_clock_count_q = []

for model in models_q:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_q += [np.mean(data["job_elapsed(seconds)"].values)]
    instructions_retired_q += [np.mean(data["instructions_retired"].values)]
    l1_misses_q += [np.mean(data["job_l1_misses"].values)]
    l1_miss_ratio_q += [np.mean(data["job_l1_miss_ratio(%%)"].values)]
    l2_misses_q += [np.mean(data["job_l2_misses"].values)]
    l2_miss_ratio_q += [np.mean(data["job_l2_miss_ratio(%%)"].values)]
    cpu_clock_count_q += [np.mean(data["cpu_clock_count"].values)]

bandwidth_q = np.divide(l2_misses_q, job_elapsed_q)
instructions_per_second_q = np.divide(instructions_retired_q, job_elapsed_q)
instructions_per_clock_q = np.divide(instructions_retired_q, cpu_clock_count_q)

flag_sum_q = []

for model in models_q:
    data = pd.read_csv(data_folder+model+"/"+"prediction_output.csv")
    flag_sum_q += [np.sum(data["prediction_flag"].values)]

examples_q = np.zeros(np.array(flag_sum_q).shape) + 10
top_1_accuracy_q = np.divide(flag_sum_q, examples_q)

# print(size_q)
# print(job_elapsed_q)
# print(top_1_accuracy_q)
# print(l2_misses_q)
# print(bandwidth_q)

# plot size of models vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.axes(yscale = "log")
plt.bar(positions, size, width=0.3, fc='orange', label='models with full precision (fp32)')
plt.bar(positions_q, size_q, width=0.3, fc='cornflowerblue', label='models after quantization (uint8)')
plt.xticks(range(0,25), short_names, rotation=75, fontsize=8)
plt.ylim(1,1000.0)
plt.xlabel('models')
plt.ylabel('size of models (MB)')
plt.legend()
# plt.title('size of models w.r.t models')

# for a,b in zip(range(len(models)), size):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('size-models.pdf', bbox_inches='tight')

# plot inference time vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.axes(yscale = "log")
plt.bar(positions, job_elapsed, width=0.3, fc='orange', label='models with full precision (fp32)')
plt.bar(positions_q, job_elapsed_q, width=0.3, fc='cornflowerblue', label='models after quantization (uint8)')
plt.xticks(range(0,25), short_names, rotation=75, fontsize=8)
plt.ylim(0.1,10)
plt.xlabel('models')
plt.ylabel('inference time (s)')
plt.legend()
# plt.title('inference time w.r.t models')

# for a,b in zip(range(len(models)), job_elapsed):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('inference_time-models.pdf', bbox_inches='tight')

# plot instructions_retired vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.axes(yscale = "log")
plt.bar(positions, instructions_retired, width=0.3, fc='orange', label='models with full precision (fp32)')
plt.bar(positions_q, instructions_retired_q, width=0.3, fc='cornflowerblue', label='models after quantization (uint8)')
plt.xticks(range(0,25), short_names, rotation=75, fontsize=8)
plt.ylim(1e8,1.4e10)
plt.xlabel('models')
plt.ylabel('retired instructions')
plt.legend()
# plt.title('retired instructions w.r.t models')

# for a,b in zip(range(len(models)), instructions_retired):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=6)

plt.savefig('instructions_retired-models.pdf', bbox_inches='tight')

# plot instructions_per_second vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.bar(positions, instructions_per_second, width=0.3, fc='orange', label='models with full precision (fp32)')
plt.bar(positions_q, instructions_per_second_q, width=0.3, fc='cornflowerblue', label='models after quantization (uint8)')
plt.xticks(range(0,25), short_names, rotation=75, fontsize=8)
plt.ylim(1.10e9,1.55e9)
plt.xlabel('models')
plt.ylabel('instructions per second')
plt.legend()
# plt.title('instructions per second w.r.t models')

# for a,b in zip(range(len(models)), instructions_per_second):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=6)

plt.savefig('instructions_per_second-models.pdf', bbox_inches='tight')

# plot instructions_per_clock vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.bar(positions, instructions_per_clock, width=0.3, fc='orange', label='models with full precision (fp32)')
plt.bar(positions_q, instructions_per_clock_q, width=0.3, fc='cornflowerblue', label='models after quantization (uint8)')
plt.xticks(range(0,25), short_names, rotation=75, fontsize=8)
plt.ylim(0.70,1.05)
plt.xlabel('models')
plt.ylabel('instructions per clock (IPC)')
plt.legend()
# plt.title('instructions per clock (IPC) w.r.t models')

# for a,b in zip(range(len(models)), instructions_per_clock):
#     plt.text(a, b, '%.3f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('instructions_per_clock-models.pdf', bbox_inches='tight')

# plot top-1 accuracy vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.bar(positions, top_1_accuracy, width=0.3, fc='orange', label='models with full precision (fp32)')
plt.bar(positions_q, top_1_accuracy_q, width=0.3, fc='cornflowerblue', label='models after quantization (uint8)')
plt.xticks(range(0,25), short_names, rotation=75, fontsize=8)
plt.ylim(64,80)
plt.xlabel('models')
plt.ylabel('top-1 accuracy (%)')
plt.legend()
# plt.title('top-1 accuracy w.r.t models')

# for a,b in zip(range(len(models)), top_1_accuracy):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('top_1_accuracy-models.pdf', bbox_inches='tight')

# plot l1_misses vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.axes(yscale = "log")
plt.bar(positions, l1_misses, width=0.3, fc='orange', label='models with full precision (fp32)')
plt.bar(positions_q, l1_misses_q, width=0.3, fc='cornflowerblue', label='models after quantization (uint8)')
plt.xticks(range(0,25), short_names, rotation=75, fontsize=8)
plt.ylim(2e5,1e8)
plt.xlabel('models')
plt.ylabel('l1-d misses')
plt.legend()
# plt.title('l1 misses w.r.t models')

# for a,b in zip(range(len(models)), l1_misses):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=6)

plt.savefig('l1_misses-models.pdf', bbox_inches='tight')

# plot l1_miss_ratio vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.bar(positions, l1_miss_ratio, width=0.3, fc='orange', label='models with full precision (fp32)')
plt.bar(positions_q, l1_miss_ratio_q, width=0.3, fc='cornflowerblue', label='models after quantization (uint8)')
plt.xticks(range(0,25), short_names, rotation=75, fontsize=8)
plt.ylim(0.002,0.018)
plt.xlabel('models')
plt.ylabel('l1-d miss ratio')
plt.legend()
# plt.title('l1 miss ratio w.r.t models')

# for a,b in zip(range(len(models)), l1_miss_ratio):
#     plt.text(a, b, '%.3f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('l1_miss_ratio-models.pdf', bbox_inches='tight')

# plot l2_misses vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.axes(yscale = "log")
plt.bar(positions, l2_misses, width=0.3, fc='orange', label='models with full precision (fp32)')
plt.bar(positions_q, l2_misses_q, width=0.3, fc='cornflowerblue', label='models after quantization (uint8)')
plt.xticks(range(0,25), short_names, rotation=75, fontsize=8)
# plt.ylim(2e5,2e8)
plt.xlabel('models')
plt.ylabel('l2 misses')
plt.legend()
# plt.title('l2 misses w.r.t models')

# for a,b in zip(range(len(models)), l2_misses):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=6)

plt.savefig('l2_misses-models.pdf', bbox_inches='tight')

# plot l2_miss_ratio vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.bar(positions, l2_miss_ratio, width=0.3, fc='orange', label='models with full precision (fp32)')
plt.bar(positions_q, l2_miss_ratio_q, width=0.3, fc='cornflowerblue', label='models after quantization (uint8)')
plt.xticks(range(0,25), short_names, rotation=75, fontsize=8)
plt.ylim(0.015,0.220)
plt.xlabel('models')
plt.ylabel('l2 miss ratio')
plt.legend()
# plt.title('l2 miss ratio w.r.t models')

# for a,b in zip(range(len(models)), l2_miss_ratio):
#     plt.text(a, b, '%.3f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('l2_miss_ratio-models.pdf', bbox_inches='tight')

# plot bandwidth vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.bar(positions, bandwidth, width=0.3, fc='orange', label='models with full precision (fp32)')
plt.bar(positions_q, bandwidth_q, width=0.3, fc='cornflowerblue', label='models after quantization (uint8)')
plt.xticks(range(0,25), short_names, rotation=75, fontsize=8)
# plt.ylim(0.4e7,2.0e7)
plt.xlabel('models')
plt.ylabel('bandwidth (64 bytes/s)')
plt.legend()
# plt.title('bandwidth w.r.t models')

# for a,b in zip(range(len(models)), bandwidth):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=6)

plt.savefig('bandwidth-models.pdf', bbox_inches='tight')

# # print(top_1_accuracy)
# print(job_elapsed)
# print(job_elapsed_q)
# # print(l2_misses)
# # print(bandwidth)
