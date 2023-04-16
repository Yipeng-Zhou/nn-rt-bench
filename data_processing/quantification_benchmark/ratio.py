from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# extract the data from the file "extra_benchmarks_fp32.csv"
extra_data = pd.read_csv("extra_benchmarks_fp32.csv")
models_fp32 = extra_data["model"].values
size_fp32 = extra_data["size(MB)"].values
short_name_fp32 = extra_data["short_name"].values

# extract the data from the folder "benchmarks"
data_folder = "benchmarks/"

job_elapsed_fp32 = []
instructions_retired_fp32 = []
l1_misses_fp32 = []
l1_miss_ratio_fp32 = []
l2_misses_fp32 = []
l2_miss_ratio_fp32 = []
cpu_clock_count_fp32 = []

for model in models_fp32:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_fp32 += [np.mean(data["job_elapsed(seconds)"].values)]
    instructions_retired_fp32 += [np.mean(data["instructions_retired"].values)]
    l1_misses_fp32 += [np.mean(data["job_l1_misses"].values)]
    l1_miss_ratio_fp32 += [np.mean(data["job_l1_miss_ratio(%%)"].values)]
    l2_misses_fp32 += [np.mean(data["job_l2_misses"].values)]
    l2_miss_ratio_fp32 += [np.mean(data["job_l2_miss_ratio(%%)"].values)]
    cpu_clock_count_fp32 += [np.mean(data["cpu_clock_count"].values)]

bandwidth_fp32 = np.divide(l2_misses_fp32, job_elapsed_fp32)
instructions_per_second_fp32 = np.divide(instructions_retired_fp32, job_elapsed_fp32)
instructions_per_clock_fp32 = np.divide(instructions_retired_fp32, cpu_clock_count_fp32)

flag_sum_fp32 = []

for model in models_fp32:
    data = pd.read_csv(data_folder+model+"/"+"prediction_output.csv")
    flag_sum_fp32 += [np.sum(data["prediction_flag"].values)]

examples_fp32 = np.zeros(np.array(flag_sum_fp32).shape) + 10
top_1_accuracy_fp32 = np.divide(flag_sum_fp32, examples_fp32)

# extract the data from the file "extra_benchmarks_uint8.csv"
extra_data = pd.read_csv("extra_benchmarks_uint8.csv")
models_uint8 = extra_data["model"].values
size_uint8 = extra_data["size(MB)"].values
short_name_uint8 = extra_data["short_name"].values

# extract the data from the folder "benchmarks"
data_folder = "benchmarks/"

job_elapsed_uint8 = []
instructions_retired_uint8 = []
l1_misses_uint8 = []
l1_miss_ratio_uint8 = []
l2_misses_uint8 = []
l2_miss_ratio_uint8 = []
cpu_clock_count_uint8 = []

for model in models_uint8:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_uint8 += [np.mean(data["job_elapsed(seconds)"].values)]
    instructions_retired_uint8 += [np.mean(data["instructions_retired"].values)]
    l1_misses_uint8 += [np.mean(data["job_l1_misses"].values)]
    l1_miss_ratio_uint8 += [np.mean(data["job_l1_miss_ratio(%%)"].values)]
    l2_misses_uint8 += [np.mean(data["job_l2_misses"].values)]
    l2_miss_ratio_uint8 += [np.mean(data["job_l2_miss_ratio(%%)"].values)]
    cpu_clock_count_uint8 += [np.mean(data["cpu_clock_count"].values)]

bandwidth_uint8 = np.divide(l2_misses_uint8, job_elapsed_uint8)
instructions_per_second_uint8 = np.divide(instructions_retired_uint8, job_elapsed_uint8)
instructions_per_clock_uint8 = np.divide(instructions_retired_uint8, cpu_clock_count_uint8)

flag_sum_uint8 = []

for model in models_uint8:
    data = pd.read_csv(data_folder+model+"/"+"prediction_output.csv")
    flag_sum_uint8 += [np.sum(data["prediction_flag"].values)]

examples_uint8 = np.zeros(np.array(flag_sum_uint8).shape) + 10
top_1_accuracy_uint8 = np.divide(flag_sum_uint8, examples_uint8)

# caculate ratio of uint8 to fp32
percent = [100, 100, 100, 100, 100, 100, 100, 100]

size_ratio = np.multiply(np.divide(size_uint8, size_fp32), percent)
inference_time_ratio = np.multiply(np.divide(job_elapsed_uint8, job_elapsed_fp32), percent)
accuracy_ratio = np.multiply(np.divide(top_1_accuracy_uint8, top_1_accuracy_fp32), percent)
l2_misses_ratio = np.multiply(np.divide(l2_misses_uint8, l2_misses_fp32), percent)
bandwidth_ratio = np.multiply(np.divide(bandwidth_uint8, bandwidth_fp32), percent)

print('size ratio')
print(size_ratio)
print('inference time ratio')
print(inference_time_ratio)
print('accuracy ratio')
print(accuracy_ratio)
print('l2 misses ratio')
print(l2_misses_ratio)
print('bandwidth ratio')
print(bandwidth_ratio)

