from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# extract the data from the file "extra_benchmarks.csv"
extra_data = pd.read_csv("extra_benchmarks_fp32.csv")
models = extra_data["model"].values
size = extra_data["size(MB)"].values
short_name = extra_data["short_name"].values

# positions = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

# extract the data from the file "extra_benchmarks_XNNPACK.csv"
extra_data = pd.read_csv("extra_benchmarks_fp32_XNNPACK.csv")
models_x = extra_data["model"].values
size_x = extra_data["size(MB)"].values
short_name_x = extra_data["short_name"].values

# positions_x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]

# extract the data from the folder "benchmarks"
data_folder = "benchmarks/"

## without XNNPACK
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

## with XNNPACK
job_elapsed_x = []
instructions_retired_x = []
l1_misses_x = []
l1_miss_ratio_x = []
l2_misses_x = []
l2_miss_ratio_x = []
cpu_clock_count_x = []

for model in models_x:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_x += [np.mean(data["job_elapsed(seconds)"].values)]
    instructions_retired_x += [np.mean(data["instructions_retired"].values)]
    l1_misses_x += [np.mean(data["job_l1_misses"].values)]
    l1_miss_ratio_x += [np.mean(data["job_l1_miss_ratio(%%)"].values)]
    l2_misses_x += [np.mean(data["job_l2_misses"].values)]
    l2_miss_ratio_x += [np.mean(data["job_l2_miss_ratio(%%)"].values)]
    cpu_clock_count_x += [np.mean(data["cpu_clock_count"].values)]

bandwidth_x = np.divide(l2_misses_x, job_elapsed_x)
instructions_per_second_x = np.divide(instructions_retired_x, job_elapsed_x)
instructions_per_clock_x = np.divide(instructions_retired_x, cpu_clock_count_x)

flag_sum_x = []

for model in models_x:
    data = pd.read_csv(data_folder+model+"/"+"prediction_output.csv")
    flag_sum_x += [np.sum(data["prediction_flag"].values)]

examples_x = np.zeros(np.array(flag_sum_x).shape) + 10
top_1_accuracy_x = np.divide(flag_sum_x, examples_x)

# print(size_x)
# print(job_elapsed_x)
# print(top_1_accuracy_x)
# print(l2_misses_x)
# print(bandwidth_x)

# import sys
# sys.exit()

# "enable XNNPACK for inference" / "inference without XNNPACK"
size_divide = np.divide(size_x, size)
instructions_retired_divide = np.divide(instructions_retired_x, instructions_retired)
job_elapsed_divide = np.divide(job_elapsed_x, job_elapsed)
instructions_per_clock_divide = np.divide(instructions_per_clock_x, instructions_per_clock)
instructions_per_second_divide = np.divide(instructions_per_second_x, instructions_per_second)
top_1_accuracy_divide = np.divide(top_1_accuracy_x, top_1_accuracy)
l1_misses_divide = np.divide(l1_misses_x, l1_misses)
l1_miss_ratio_divide = np.divide(l1_miss_ratio_x, l1_miss_ratio)
l2_misses_divide = np.divide(l2_misses_x, l2_misses)
l2_miss_ratio_divide = np.divide(l2_miss_ratio_x, l2_miss_ratio)
bandwidth_divide = np.divide(bandwidth_x, bandwidth)

# # percent
# percent = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
# instructions_retired_divide = np.multiply(instructions_retired_divide, percent)
# job_elapsed_divide = np.multiply(job_elapsed_divide, percent)
# l2_misses_divide = np.multiply(l2_misses_divide, percent)
# bandwidth_divide = np.multiply(bandwidth_divide, percent)
# instructions_per_clock_divide = np.multiply(instructions_per_clock_divide, percent)

# print('instructions retired')
# print(np.around(instructions_retired_divide,2))
# print('inference time')
# print(np.around(job_elapsed_divide,2))
# print('l2 misses')
# print(np.around(l2_misses_divide,2))
# print('bandwidth')
# print(np.around(bandwidth_divide,2))
# print('instructions per clock')
# print(np.around(instructions_per_clock_divide,2))

# print(np.mean(job_elapsed_divide))

# import sys
# sys.exit()

# plot size of models vs. models
plt.figure(figsize=(15,6), dpi=120)
# plt.axes(yscale = "log")
plt.bar(range(17), size_divide, width=0.3, fc='orange', label='enable XNNPACK for inference / inference without XNNPACK')
plt.xticks(range(17), short_name, rotation=75, fontsize=8)
plt.ylim(0.8,1.1)
plt.plot([-0.5,16.5],[1,1], linestyle='--', color='cornflowerblue')
plt.xlabel('models')
plt.ylabel('ratio of model size')
plt.legend()
# plt.title('size of models w.r.t models')

# for a,b in zip(range(len(models)), size):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('size-models_XNNPACK.pdf', bbox_inches='tight')

# plot inference time vs. models
plt.figure(figsize=(15,6), dpi=120)
# plt.axes(yscale = "log")
plt.bar(range(17), job_elapsed_divide, width=0.3, fc='orange', label='enable XNNPACK for inference / inference without XNNPACK')
plt.xticks(range(17), short_name, rotation=75, fontsize=8)
plt.ylim(0.65,0.85)
# plt.plot([-0.5,16.5],[1,1], linestyle='--', color='cornflowerblue')
plt.xlabel('models')
plt.ylabel('ratio of inference time')
plt.legend()
# plt.title('inference time w.r.t models')

# for a,b in zip(range(len(models)), job_elapsed):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('inference_time-models_XNNPACK.pdf', bbox_inches='tight')

# plot instructions_retired vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.bar(range(17), instructions_retired_divide, width=0.3, fc='orange', label='enable XNNPACK for inference / inference without XNNPACK')
plt.xticks(range(17), short_name, rotation=75, fontsize=8)
plt.ylim(0.9,1.0)
# plt.plot([-0.5,16.5],[1,1], linestyle='--', color='cornflowerblue')
plt.xlabel('models')
plt.ylabel('ratio of retired instructions')
plt.legend()
# plt.title('retired instructions w.r.t models')

# for a,b in zip(range(len(models)), instructions_retired):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=6)

plt.savefig('instructions_retired-models_XNNPACK.pdf', bbox_inches='tight')

# plot instructions_per_second vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.bar(range(17), instructions_per_second_divide, width=0.3, fc='orange', label='enable XNNPACK for inference / inference without XNNPACK')
plt.xticks(range(17), short_name, rotation=75, fontsize=8)
plt.ylim(1.1,1.4)
# plt.plot([-0.5,16.5],[1,1], linestyle='--', color='cornflowerblue')
plt.xlabel('models')
plt.ylabel('ratio of instructions per second')
plt.legend()
# plt.title('instructions per second w.r.t models')

# for a,b in zip(range(len(models)), instructions_per_second):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=6)

plt.savefig('instructions_per_second-models_XNNPACK.pdf', bbox_inches='tight')

# plot instructions_per_clock vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.bar(range(17), instructions_per_clock_divide, width=0.3, fc='orange', label='enable XNNPACK for inference / inference without XNNPACK')
plt.xticks(range(17), short_name, rotation=75, fontsize=8)
plt.ylim(1.1,1.4)
# plt.plot([-0.5,16.5],[1,1], linestyle='--', color='cornflowerblue')
plt.xlabel('models')
plt.ylabel('ratio of instructions per clock (IPC)')
plt.legend()
# plt.title('instructions per clock (IPC) w.r.t models')

# for a,b in zip(range(len(models)), instructions_per_clock):
#     plt.text(a, b, '%.3f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('instructions_per_clock-models_XNNPACK.pdf', bbox_inches='tight')

# plot top-1 accuracy vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.bar(range(17), top_1_accuracy_divide, width=0.3, fc='orange', label='enable XNNPACK for inference / inference without XNNPACK')
plt.xticks(range(17), short_name, rotation=75, fontsize=8)
plt.ylim(0.8,1.1)
plt.plot([-0.5,16.5],[1,1], linestyle='--', color='cornflowerblue')
plt.xlabel('models')
plt.ylabel('ratio of top-1 accuracy')
plt.legend()
# plt.title('top-1 accuracy w.r.t models')

# for a,b in zip(range(len(models)), top_1_accuracy):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('top_1_accuracy-models_XNNPACK.pdf', bbox_inches='tight')

# plot l1_misses vs. models
plt.figure(figsize=(15,6), dpi=120)
# plt.axes(yscale = "log")
plt.bar(range(17), l1_misses_divide, width=0.3, fc='orange', label='enable XNNPACK for inference / inference without XNNPACK')
plt.xticks(range(17), short_name, rotation=75, fontsize=8)
plt.ylim(0.55,1.65)
plt.plot([-0.5,16.5],[1,1], linestyle='--', color='cornflowerblue')
plt.xlabel('models')
plt.ylabel('ratio of l1-d misses')
plt.legend()
# plt.title('l1 misses w.r.t models')

# for a,b in zip(range(len(models)), l1_misses):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=6)

plt.savefig('l1_misses-models_XNNPACK.pdf', bbox_inches='tight')

# plot l1_miss_ratio vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.bar(range(17), l1_miss_ratio_divide, width=0.3, fc='orange', label='enable XNNPACK for inference / inference without XNNPACK')
plt.xticks(range(17), short_name, rotation=75, fontsize=8)
plt.ylim(0.5,1.6)
plt.plot([-0.5,16.5],[1,1], linestyle='--', color='cornflowerblue')
plt.xlabel('models')
plt.ylabel('ratio of "l1-d miss ratio"')
plt.legend()
# plt.title('l1 miss ratio w.r.t models')

# for a,b in zip(range(len(models)), l1_miss_ratio):
#     plt.text(a, b, '%.3f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('l1_miss_ratio-models_XNNPACK.pdf', bbox_inches='tight')

# plot l2_misses vs. models
plt.figure(figsize=(15,6), dpi=120)
# plt.axes(yscale = "log")
plt.bar(range(17), l2_misses_divide, width=0.3, fc='orange', label='enable XNNPACK for inference / inference without XNNPACK')
plt.xticks(range(17), short_name, rotation=75, fontsize=8)
plt.ylim(0.6,1.7)
plt.plot([-0.5,16.5],[1,1], linestyle='--', color='cornflowerblue')
plt.xlabel('models')
plt.ylabel('ratio of l2 misses')
plt.legend()
# plt.title('l2 misses w.r.t models')

# for a,b in zip(range(len(models)), l2_misses):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=6)

plt.savefig('l2_misses-models_XNNPACK.pdf', bbox_inches='tight')

# plot l2_miss_ratio vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.bar(range(17), l2_miss_ratio_divide, width=0.3, fc='orange', label='enable XNNPACK for inference / inference without XNNPACK')
plt.xticks(range(17), short_name, rotation=75, fontsize=8)
plt.ylim(0.8,2.0)
plt.plot([-0.5,16.5],[1,1], linestyle='--', color='cornflowerblue')
plt.xlabel('models')
plt.ylabel('ratio of "l2 miss ratio"')
plt.legend()
# plt.title('l2 miss ratio w.r.t models')

# for a,b in zip(range(len(models)), l2_miss_ratio):
#     plt.text(a, b, '%.3f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('l2_miss_ratio-models_XNNPACK.pdf', bbox_inches='tight')

# plot bandwidth vs. models
plt.figure(figsize=(15,6), dpi=120)
plt.bar(range(17), bandwidth_divide, width=0.3, fc='orange', label='enable XNNPACK for inference / inference without XNNPACK')
plt.xticks(range(17), short_name, rotation=75, fontsize=8)
plt.ylim(1.0,2.3)
plt.plot([-0.5,16.5],[1,1], linestyle='--', color='cornflowerblue')
plt.xlabel('models')
plt.ylabel('ratio of bandwidth')
plt.legend()
# plt.title('bandwidth w.r.t models')

# for a,b in zip(range(len(models)), bandwidth):
#     plt.text(a, b+0.03, '%.3f'%b, ha='center', va='bottom', fontsize=6)

plt.savefig('bandwidth-models_XNNPACK.pdf', bbox_inches='tight')
