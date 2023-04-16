from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colors = ["lightgreen", "cornflowerblue", "mediumorchid", "orange", "hotpink"]
labels = ["VGG family",
          "MobileNet family", 
          "EfficientNet family",
          "Inception family",
          "ResNet family"]

def add_label(add_labels, violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    add_labels.append((mpatches.Patch(color=color), label))

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

# extract the data from the file "extra_benchmarks_fp32_xxx_XNNPACK.csv"
extra_data = pd.read_csv("extra_benchmarks_fp32_vgg_XNNPACK.csv")
models_fp32_vgg_XNNPACK = extra_data["model"].values
size_fp32_vgg_XNNPACK = extra_data["size(MB)"].values
short_name_fp32_vgg_XNNPACK = extra_data["short_name"].values

extra_data = pd.read_csv("extra_benchmarks_fp32_mobilenet_XNNPACK.csv")
models_fp32_mobilenet_XNNPACK = extra_data["model"].values
size_fp32_mobilenet_XNNPACK = extra_data["size(MB)"].values
short_name_fp32_mobilenet_XNNPACK = extra_data["short_name"].values

extra_data = pd.read_csv("extra_benchmarks_fp32_efficientnet_XNNPACK.csv")
models_fp32_efficientnet_XNNPACK = extra_data["model"].values
size_fp32_efficientnet_XNNPACK = extra_data["size(MB)"].values
short_name_fp32_efficientnet_XNNPACK = extra_data["short_name"].values

extra_data = pd.read_csv("extra_benchmarks_fp32_inception_XNNPACK.csv")
models_fp32_inception_XNNPACK = extra_data["model"].values
size_fp32_inception_XNNPACK = extra_data["size(MB)"].values
short_name_fp32_inception_XNNPACK = extra_data["short_name"].values

extra_data = pd.read_csv("extra_benchmarks_fp32_resnet_XNNPACK.csv")
models_fp32_resnet_XNNPACK = extra_data["model"].values
size_fp32_resnet_XNNPACK = extra_data["size(MB)"].values
short_name_fp32_resnet_XNNPACK = extra_data["short_name"].values

# extract the short names from "extra_benchmarks_fp32_XNNPACK.csv"
extra_data = pd.read_csv("extra_benchmarks_fp32_XNNPACK.csv")
short_name_fp32_XNNPACK = extra_data["short_name"].values

# extract the data from the folder "benchmarks"
data_folder = "benchmarks/"

## fp32 vgg XNNPACK
job_elapsed_fp32_vgg_XNNPACK = np.zeros((1000,2))
instructions_retired_fp32_vgg_XNNPACK = np.zeros((1000,2))
l1_misses_fp32_vgg_XNNPACK = np.zeros((1000,2))
l1_miss_ratio_fp32_vgg_XNNPACK = np.zeros((1000,2))
l2_misses_fp32_vgg_XNNPACK = np.zeros((1000,2))
l2_miss_ratio_fp32_vgg_XNNPACK = np.zeros((1000,2))
cpu_clock_count_fp32_vgg_XNNPACK = np.zeros((1000,2))

bandwidth_fp32_vgg_XNNPACK = np.zeros((1000,2))
instructions_per_second_fp32_vgg_XNNPACK = np.zeros((1000,2))
instructions_per_clock_fp32_vgg_XNNPACK = np.zeros((1000,2))

index = 0
for model in models_fp32_vgg_XNNPACK:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_fp32_vgg_XNNPACK[:,index] = np.array(data["job_elapsed(seconds)"].values)
    instructions_retired_fp32_vgg_XNNPACK[:,index] = np.array(data["instructions_retired"].values)
    l1_misses_fp32_vgg_XNNPACK[:,index] = np.array(data["job_l1_misses"].values)
    l1_miss_ratio_fp32_vgg_XNNPACK[:,index] = np.array(data["job_l1_miss_ratio(%%)"].values)
    l2_misses_fp32_vgg_XNNPACK[:,index] = np.array(data["job_l2_misses"].values)
    l2_miss_ratio_fp32_vgg_XNNPACK[:,index] = np.array(data["job_l2_miss_ratio(%%)"].values)
    cpu_clock_count_fp32_vgg_XNNPACK[:,index] = np.array(data["cpu_clock_count"].values)

    bandwidth_fp32_vgg_XNNPACK[:,index] = np.divide(l2_misses_fp32_vgg_XNNPACK[:,index], job_elapsed_fp32_vgg_XNNPACK[:,index])
    instructions_per_second_fp32_vgg_XNNPACK[:,index] = np.divide(instructions_retired_fp32_vgg_XNNPACK[:,index], job_elapsed_fp32_vgg_XNNPACK[:,index])
    instructions_per_clock_fp32_vgg_XNNPACK[:,index] = np.divide(instructions_retired_fp32_vgg_XNNPACK[:,index], cpu_clock_count_fp32_vgg_XNNPACK[:,index])

    index = index + 1

job_elapsed_fp32_vgg_XNNPACK[:,0] = job_elapsed_fp32_vgg_XNNPACK[:,0] / job_elapsed_fp32[0]
job_elapsed_fp32_vgg_XNNPACK[:,1] = job_elapsed_fp32_vgg_XNNPACK[:,1] / job_elapsed_fp32[1]
instructions_retired_fp32_vgg_XNNPACK[:,0] = instructions_retired_fp32_vgg_XNNPACK[:,0] / instructions_retired_fp32[0]
instructions_retired_fp32_vgg_XNNPACK[:,1] = instructions_retired_fp32_vgg_XNNPACK[:,1] / instructions_retired_fp32[1]
l1_misses_fp32_vgg_XNNPACK[:,0] = l1_misses_fp32_vgg_XNNPACK[:,0] / l1_misses_fp32[0]
l1_misses_fp32_vgg_XNNPACK[:,1] = l1_misses_fp32_vgg_XNNPACK[:,1] / l1_misses_fp32[1]
l2_misses_fp32_vgg_XNNPACK[:,0] = l2_misses_fp32_vgg_XNNPACK[:,0] / l2_misses_fp32[0]
l2_misses_fp32_vgg_XNNPACK[:,1] = l2_misses_fp32_vgg_XNNPACK[:,1] / l2_misses_fp32[1]


position_fp32_vgg_XNNPACK = [0, 1]

## fp32 mobilenet XNNPACK
job_elapsed_fp32_mobilenet_XNNPACK = np.zeros((1000,3))
instructions_retired_fp32_mobilenet_XNNPACK = np.zeros((1000,3))
l1_misses_fp32_mobilenet_XNNPACK = np.zeros((1000,3))
l1_miss_ratio_fp32_mobilenet_XNNPACK = np.zeros((1000,3))
l2_misses_fp32_mobilenet_XNNPACK = np.zeros((1000,3))
l2_miss_ratio_fp32_mobilenet_XNNPACK = np.zeros((1000,3))
cpu_clock_count_fp32_mobilenet_XNNPACK = np.zeros((1000,3))

bandwidth_fp32_mobilenet_XNNPACK = np.zeros((1000,3))
instructions_per_second_fp32_mobilenet_XNNPACK = np.zeros((1000,3))
instructions_per_clock_fp32_mobilenet_XNNPACK = np.zeros((1000,3))

index = 0
for model in models_fp32_mobilenet_XNNPACK:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_fp32_mobilenet_XNNPACK[:,index] = np.array(data["job_elapsed(seconds)"].values)
    instructions_retired_fp32_mobilenet_XNNPACK[:,index] = np.array(data["instructions_retired"].values)
    l1_misses_fp32_mobilenet_XNNPACK[:,index] = np.array(data["job_l1_misses"].values)
    l1_miss_ratio_fp32_mobilenet_XNNPACK[:,index] = np.array(data["job_l1_miss_ratio(%%)"].values)
    l2_misses_fp32_mobilenet_XNNPACK[:,index] = np.array(data["job_l2_misses"].values)
    l2_miss_ratio_fp32_mobilenet_XNNPACK[:,index] = np.array(data["job_l2_miss_ratio(%%)"].values)
    cpu_clock_count_fp32_mobilenet_XNNPACK[:,index] = np.array(data["cpu_clock_count"].values)

    bandwidth_fp32_mobilenet_XNNPACK[:,index] = np.divide(l2_misses_fp32_mobilenet_XNNPACK[:,index], job_elapsed_fp32_mobilenet_XNNPACK[:,index])
    instructions_per_second_fp32_mobilenet_XNNPACK[:,index] = np.divide(instructions_retired_fp32_mobilenet_XNNPACK[:,index], job_elapsed_fp32_mobilenet_XNNPACK[:,index])
    instructions_per_clock_fp32_mobilenet_XNNPACK[:,index] = np.divide(instructions_retired_fp32_mobilenet_XNNPACK[:,index], cpu_clock_count_fp32_mobilenet_XNNPACK[:,index])

    index = index + 1

job_elapsed_fp32_mobilenet_XNNPACK[:,0] = job_elapsed_fp32_mobilenet_XNNPACK[:,0] / job_elapsed_fp32[2]
job_elapsed_fp32_mobilenet_XNNPACK[:,1] = job_elapsed_fp32_mobilenet_XNNPACK[:,1] / job_elapsed_fp32[3]
job_elapsed_fp32_mobilenet_XNNPACK[:,2] = job_elapsed_fp32_mobilenet_XNNPACK[:,2] / job_elapsed_fp32[4]
instructions_retired_fp32_mobilenet_XNNPACK[:,0] = instructions_retired_fp32_mobilenet_XNNPACK[:,0] / instructions_retired_fp32[2]
instructions_retired_fp32_mobilenet_XNNPACK[:,1] = instructions_retired_fp32_mobilenet_XNNPACK[:,1] / instructions_retired_fp32[3]
instructions_retired_fp32_mobilenet_XNNPACK[:,2] = instructions_retired_fp32_mobilenet_XNNPACK[:,2] / instructions_retired_fp32[4]
l1_misses_fp32_mobilenet_XNNPACK[:,0] = l1_misses_fp32_mobilenet_XNNPACK[:,0] / l1_misses_fp32[2]
l1_misses_fp32_mobilenet_XNNPACK[:,1] = l1_misses_fp32_mobilenet_XNNPACK[:,1] / l1_misses_fp32[3]
l1_misses_fp32_mobilenet_XNNPACK[:,2] = l1_misses_fp32_mobilenet_XNNPACK[:,2] / l1_misses_fp32[4]
l2_misses_fp32_mobilenet_XNNPACK[:,0] = l2_misses_fp32_mobilenet_XNNPACK[:,0] / l2_misses_fp32[2]
l2_misses_fp32_mobilenet_XNNPACK[:,1] = l2_misses_fp32_mobilenet_XNNPACK[:,1] / l2_misses_fp32[3]
l2_misses_fp32_mobilenet_XNNPACK[:,2] = l2_misses_fp32_mobilenet_XNNPACK[:,2] / l2_misses_fp32[4]

position_fp32_mobilenet_XNNPACK = [2, 3, 4]

## fp32 efficientnet XNNPACK
job_elapsed_fp32_efficientnet_XNNPACK = np.zeros((1000,1))
instructions_retired_fp32_efficientnet_XNNPACK = np.zeros((1000,1))
l1_misses_fp32_efficientnet_XNNPACK = np.zeros((1000,1))
l1_miss_ratio_fp32_efficientnet_XNNPACK = np.zeros((1000,1))
l2_misses_fp32_efficientnet_XNNPACK = np.zeros((1000,1))
l2_miss_ratio_fp32_efficientnet_XNNPACK = np.zeros((1000,1))
cpu_clock_count_fp32_efficientnet_XNNPACK = np.zeros((1000,1))

bandwidth_fp32_efficientnet_XNNPACK = np.zeros((1000,1))
instructions_per_second_fp32_efficientnet_XNNPACK = np.zeros((1000,1))
instructions_per_clock_fp32_efficientnet_XNNPACK = np.zeros((1000,1))

index = 0
for model in models_fp32_efficientnet_XNNPACK:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_fp32_efficientnet_XNNPACK[:,index] = np.array(data["job_elapsed(seconds)"].values)
    instructions_retired_fp32_efficientnet_XNNPACK[:,index] = np.array(data["instructions_retired"].values)
    l1_misses_fp32_efficientnet_XNNPACK[:,index] = np.array(data["job_l1_misses"].values)
    l1_miss_ratio_fp32_efficientnet_XNNPACK[:,index] = np.array(data["job_l1_miss_ratio(%%)"].values)
    l2_misses_fp32_efficientnet_XNNPACK[:,index] = np.array(data["job_l2_misses"].values)
    l2_miss_ratio_fp32_efficientnet_XNNPACK[:,index] = np.array(data["job_l2_miss_ratio(%%)"].values)
    cpu_clock_count_fp32_efficientnet_XNNPACK[:,index] = np.array(data["cpu_clock_count"].values)

    bandwidth_fp32_efficientnet_XNNPACK[:,index] = np.divide(l2_misses_fp32_efficientnet_XNNPACK[:,index], job_elapsed_fp32_efficientnet_XNNPACK[:,index])
    instructions_per_second_fp32_efficientnet_XNNPACK[:,index] = np.divide(instructions_retired_fp32_efficientnet_XNNPACK[:,index], job_elapsed_fp32_efficientnet_XNNPACK[:,index])
    instructions_per_clock_fp32_efficientnet_XNNPACK[:,index] = np.divide(instructions_retired_fp32_efficientnet_XNNPACK[:,index], cpu_clock_count_fp32_efficientnet_XNNPACK[:,index])

    index = index + 1

job_elapsed_fp32_efficientnet_XNNPACK[:,0] = job_elapsed_fp32_efficientnet_XNNPACK[:,0] / job_elapsed_fp32[5]
instructions_retired_fp32_efficientnet_XNNPACK[:,0] = instructions_retired_fp32_efficientnet_XNNPACK[:,0] / instructions_retired_fp32[5]
l1_misses_fp32_efficientnet_XNNPACK[:,0] = l1_misses_fp32_efficientnet_XNNPACK[:,0] / l1_misses_fp32[5]
l2_misses_fp32_efficientnet_XNNPACK[:,0] = l2_misses_fp32_efficientnet_XNNPACK[:,0] / l2_misses_fp32[5]

position_fp32_efficientnet_XNNPACK = [5]

## fp32 inception XNNPACK
job_elapsed_fp32_inception_XNNPACK = np.zeros((1000,5))
instructions_retired_fp32_inception_XNNPACK = np.zeros((1000,5))
l1_misses_fp32_inception_XNNPACK = np.zeros((1000,5))
l1_miss_ratio_fp32_inception_XNNPACK = np.zeros((1000,5))
l2_misses_fp32_inception_XNNPACK = np.zeros((1000,5))
l2_miss_ratio_fp32_inception_XNNPACK = np.zeros((1000,5))
cpu_clock_count_fp32_inception_XNNPACK = np.zeros((1000,5))

bandwidth_fp32_inception_XNNPACK = np.zeros((1000,5))
instructions_per_second_fp32_inception_XNNPACK = np.zeros((1000,5))
instructions_per_clock_fp32_inception_XNNPACK = np.zeros((1000,5))

index = 0
for model in models_fp32_inception_XNNPACK:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_fp32_inception_XNNPACK[:,index] = np.array(data["job_elapsed(seconds)"].values)
    instructions_retired_fp32_inception_XNNPACK[:,index] = np.array(data["instructions_retired"].values)
    l1_misses_fp32_inception_XNNPACK[:,index] = np.array(data["job_l1_misses"].values)
    l1_miss_ratio_fp32_inception_XNNPACK[:,index] = np.array(data["job_l1_miss_ratio(%%)"].values)
    l2_misses_fp32_inception_XNNPACK[:,index] = np.array(data["job_l2_misses"].values)
    l2_miss_ratio_fp32_inception_XNNPACK[:,index] = np.array(data["job_l2_miss_ratio(%%)"].values)
    cpu_clock_count_fp32_inception_XNNPACK[:,index] = np.array(data["cpu_clock_count"].values)

    bandwidth_fp32_inception_XNNPACK[:,index] = np.divide(l2_misses_fp32_inception_XNNPACK[:,index], job_elapsed_fp32_inception_XNNPACK[:,index])
    instructions_per_second_fp32_inception_XNNPACK[:,index] = np.divide(instructions_retired_fp32_inception_XNNPACK[:,index], job_elapsed_fp32_inception_XNNPACK[:,index])
    instructions_per_clock_fp32_inception_XNNPACK[:,index] = np.divide(instructions_retired_fp32_inception_XNNPACK[:,index], cpu_clock_count_fp32_inception_XNNPACK[:,index])

    index = index + 1   

job_elapsed_fp32_inception_XNNPACK[:,0] = job_elapsed_fp32_inception_XNNPACK[:,0] / job_elapsed_fp32[6]
job_elapsed_fp32_inception_XNNPACK[:,1] = job_elapsed_fp32_inception_XNNPACK[:,1] / job_elapsed_fp32[7]
job_elapsed_fp32_inception_XNNPACK[:,2] = job_elapsed_fp32_inception_XNNPACK[:,2] / job_elapsed_fp32[8]
job_elapsed_fp32_inception_XNNPACK[:,3] = job_elapsed_fp32_inception_XNNPACK[:,3] / job_elapsed_fp32[9]
job_elapsed_fp32_inception_XNNPACK[:,4] = job_elapsed_fp32_inception_XNNPACK[:,4] / job_elapsed_fp32[10]
instructions_retired_fp32_inception_XNNPACK[:,0] = instructions_retired_fp32_inception_XNNPACK[:,0] / instructions_retired_fp32[6]
instructions_retired_fp32_inception_XNNPACK[:,1] = instructions_retired_fp32_inception_XNNPACK[:,1] / instructions_retired_fp32[7]
instructions_retired_fp32_inception_XNNPACK[:,2] = instructions_retired_fp32_inception_XNNPACK[:,2] / instructions_retired_fp32[8]
instructions_retired_fp32_inception_XNNPACK[:,3] = instructions_retired_fp32_inception_XNNPACK[:,3] / instructions_retired_fp32[9]
instructions_retired_fp32_inception_XNNPACK[:,4] = instructions_retired_fp32_inception_XNNPACK[:,4] / instructions_retired_fp32[10]
l1_misses_fp32_inception_XNNPACK[:,0] = l1_misses_fp32_inception_XNNPACK[:,0] / l1_misses_fp32[6]
l1_misses_fp32_inception_XNNPACK[:,1] = l1_misses_fp32_inception_XNNPACK[:,1] / l1_misses_fp32[7]
l1_misses_fp32_inception_XNNPACK[:,2] = l1_misses_fp32_inception_XNNPACK[:,2] / l1_misses_fp32[8]
l1_misses_fp32_inception_XNNPACK[:,3] = l1_misses_fp32_inception_XNNPACK[:,3] / l1_misses_fp32[9]
l1_misses_fp32_inception_XNNPACK[:,4] = l1_misses_fp32_inception_XNNPACK[:,4] / l1_misses_fp32[10]
l2_misses_fp32_inception_XNNPACK[:,0] = l2_misses_fp32_inception_XNNPACK[:,0] / l2_misses_fp32[6]
l2_misses_fp32_inception_XNNPACK[:,1] = l2_misses_fp32_inception_XNNPACK[:,1] / l2_misses_fp32[7]
l2_misses_fp32_inception_XNNPACK[:,2] = l2_misses_fp32_inception_XNNPACK[:,2] / l2_misses_fp32[8]
l2_misses_fp32_inception_XNNPACK[:,3] = l2_misses_fp32_inception_XNNPACK[:,3] / l2_misses_fp32[9]
l2_misses_fp32_inception_XNNPACK[:,4] = l2_misses_fp32_inception_XNNPACK[:,4] / l2_misses_fp32[10]

position_fp32_inception_XNNPACK = [6, 7, 8, 9, 10]

## fp32 resnet XNNPACK
job_elapsed_fp32_resnet_XNNPACK = np.zeros((1000,6))
instructions_retired_fp32_resnet_XNNPACK = np.zeros((1000,6))
l1_misses_fp32_resnet_XNNPACK = np.zeros((1000,6))
l1_miss_ratio_fp32_resnet_XNNPACK = np.zeros((1000,6))
l2_misses_fp32_resnet_XNNPACK = np.zeros((1000,6))
l2_miss_ratio_fp32_resnet_XNNPACK = np.zeros((1000,6))
cpu_clock_count_fp32_resnet_XNNPACK = np.zeros((1000,6))

bandwidth_fp32_resnet_XNNPACK = np.zeros((1000,6))
instructions_per_second_fp32_resnet_XNNPACK = np.zeros((1000,6))
instructions_per_clock_fp32_resnet_XNNPACK = np.zeros((1000,6))

index = 0
for model in models_fp32_resnet_XNNPACK:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_fp32_resnet_XNNPACK[:,index] = np.array(data["job_elapsed(seconds)"].values)
    instructions_retired_fp32_resnet_XNNPACK[:,index] = np.array(data["instructions_retired"].values)
    l1_misses_fp32_resnet_XNNPACK[:,index] = np.array(data["job_l1_misses"].values)
    l1_miss_ratio_fp32_resnet_XNNPACK[:,index] = np.array(data["job_l1_miss_ratio(%%)"].values)
    l2_misses_fp32_resnet_XNNPACK[:,index] = np.array(data["job_l2_misses"].values)
    l2_miss_ratio_fp32_resnet_XNNPACK[:,index] = np.array(data["job_l2_miss_ratio(%%)"].values)
    cpu_clock_count_fp32_resnet_XNNPACK[:,index] = np.array(data["cpu_clock_count"].values)

    bandwidth_fp32_resnet_XNNPACK[:,index] = np.divide(l2_misses_fp32_resnet_XNNPACK[:,index], job_elapsed_fp32_resnet_XNNPACK[:,index])
    instructions_per_second_fp32_resnet_XNNPACK[:,index] = np.divide(instructions_retired_fp32_resnet_XNNPACK[:,index], job_elapsed_fp32_resnet_XNNPACK[:,index])
    instructions_per_clock_fp32_resnet_XNNPACK[:,index] = np.divide(instructions_retired_fp32_resnet_XNNPACK[:,index], cpu_clock_count_fp32_resnet_XNNPACK[:,index])

    index = index + 1 

job_elapsed_fp32_resnet_XNNPACK[:,0] = job_elapsed_fp32_resnet_XNNPACK[:,0] / job_elapsed_fp32[11]
job_elapsed_fp32_resnet_XNNPACK[:,1] = job_elapsed_fp32_resnet_XNNPACK[:,1] / job_elapsed_fp32[12]
job_elapsed_fp32_resnet_XNNPACK[:,2] = job_elapsed_fp32_resnet_XNNPACK[:,2] / job_elapsed_fp32[13]
job_elapsed_fp32_resnet_XNNPACK[:,3] = job_elapsed_fp32_resnet_XNNPACK[:,3] / job_elapsed_fp32[14]
job_elapsed_fp32_resnet_XNNPACK[:,4] = job_elapsed_fp32_resnet_XNNPACK[:,4] / job_elapsed_fp32[15]
job_elapsed_fp32_resnet_XNNPACK[:,5] = job_elapsed_fp32_resnet_XNNPACK[:,5] / job_elapsed_fp32[16]
instructions_retired_fp32_resnet_XNNPACK[:,0] = instructions_retired_fp32_resnet_XNNPACK[:,0] / instructions_retired_fp32[11]
instructions_retired_fp32_resnet_XNNPACK[:,1] = instructions_retired_fp32_resnet_XNNPACK[:,1] / instructions_retired_fp32[12]
instructions_retired_fp32_resnet_XNNPACK[:,2] = instructions_retired_fp32_resnet_XNNPACK[:,2] / instructions_retired_fp32[13]
instructions_retired_fp32_resnet_XNNPACK[:,3] = instructions_retired_fp32_resnet_XNNPACK[:,3] / instructions_retired_fp32[14]
instructions_retired_fp32_resnet_XNNPACK[:,4] = instructions_retired_fp32_resnet_XNNPACK[:,4] / instructions_retired_fp32[15]
instructions_retired_fp32_resnet_XNNPACK[:,5] = instructions_retired_fp32_resnet_XNNPACK[:,5] / instructions_retired_fp32[16]
l1_misses_fp32_resnet_XNNPACK[:,0] = l1_misses_fp32_resnet_XNNPACK[:,0] / l1_misses_fp32[11]
l1_misses_fp32_resnet_XNNPACK[:,1] = l1_misses_fp32_resnet_XNNPACK[:,1] / l1_misses_fp32[12]
l1_misses_fp32_resnet_XNNPACK[:,2] = l1_misses_fp32_resnet_XNNPACK[:,2] / l1_misses_fp32[13]
l1_misses_fp32_resnet_XNNPACK[:,3] = l1_misses_fp32_resnet_XNNPACK[:,3] / l1_misses_fp32[14]
l1_misses_fp32_resnet_XNNPACK[:,4] = l1_misses_fp32_resnet_XNNPACK[:,4] / l1_misses_fp32[15]
l1_misses_fp32_resnet_XNNPACK[:,5] = l1_misses_fp32_resnet_XNNPACK[:,5] / l1_misses_fp32[16]
l2_misses_fp32_resnet_XNNPACK[:,0] = l2_misses_fp32_resnet_XNNPACK[:,0] / l2_misses_fp32[11]
l2_misses_fp32_resnet_XNNPACK[:,1] = l2_misses_fp32_resnet_XNNPACK[:,1] / l2_misses_fp32[12]
l2_misses_fp32_resnet_XNNPACK[:,2] = l2_misses_fp32_resnet_XNNPACK[:,2] / l2_misses_fp32[13]
l2_misses_fp32_resnet_XNNPACK[:,3] = l2_misses_fp32_resnet_XNNPACK[:,3] / l2_misses_fp32[14]
l2_misses_fp32_resnet_XNNPACK[:,4] = l2_misses_fp32_resnet_XNNPACK[:,4] / l2_misses_fp32[15]
l2_misses_fp32_resnet_XNNPACK[:,5] = l2_misses_fp32_resnet_XNNPACK[:,5] / l2_misses_fp32[16]

position_fp32_resnet_XNNPACK = [11, 12, 13, 14, 15, 16]

# print(np.where(job_elapsed_fp32_vgg_XNNPACK[:,0]==np.max(job_elapsed_fp32_vgg_XNNPACK[:,0])))
# print(np.where(job_elapsed_fp32_vgg_XNNPACK[:,1]==np.max(job_elapsed_fp32_vgg_XNNPACK[:,1])))

# print(np.where(job_elapsed_fp32_mobilenet_XNNPACK[:,0]==np.max(job_elapsed_fp32_mobilenet_XNNPACK[:,0])))
# print(np.where(job_elapsed_fp32_mobilenet_XNNPACK[:,1]==np.max(job_elapsed_fp32_mobilenet_XNNPACK[:,1])))
# print(np.where(job_elapsed_fp32_mobilenet_XNNPACK[:,2]==np.max(job_elapsed_fp32_mobilenet_XNNPACK[:,2])))

# print(np.where(job_elapsed_fp32_efficientnet_XNNPACK[:,0]==np.max(job_elapsed_fp32_efficientnet_XNNPACK[:,0])))

# print(np.where(job_elapsed_fp32_inception_XNNPACK[:,0]==np.max(job_elapsed_fp32_inception_XNNPACK[:,0])))
# print(np.where(job_elapsed_fp32_inception_XNNPACK[:,1]==np.max(job_elapsed_fp32_inception_XNNPACK[:,1])))
# print(np.where(job_elapsed_fp32_inception_XNNPACK[:,2]==np.max(job_elapsed_fp32_inception_XNNPACK[:,2])))
# print(np.where(job_elapsed_fp32_inception_XNNPACK[:,3]==np.max(job_elapsed_fp32_inception_XNNPACK[:,3])))
# print(np.where(job_elapsed_fp32_inception_XNNPACK[:,4]==np.max(job_elapsed_fp32_inception_XNNPACK[:,4])))

# print(np.where(job_elapsed_fp32_resnet_XNNPACK[:,0]==np.max(job_elapsed_fp32_resnet_XNNPACK[:,0])))
# print(np.where(job_elapsed_fp32_resnet_XNNPACK[:,1]==np.max(job_elapsed_fp32_resnet_XNNPACK[:,1])))
# print(np.where(job_elapsed_fp32_resnet_XNNPACK[:,2]==np.max(job_elapsed_fp32_resnet_XNNPACK[:,2])))
# print(np.where(job_elapsed_fp32_resnet_XNNPACK[:,3]==np.max(job_elapsed_fp32_resnet_XNNPACK[:,3])))
# print(np.where(job_elapsed_fp32_resnet_XNNPACK[:,4]==np.max(job_elapsed_fp32_resnet_XNNPACK[:,4])))
# print(np.where(job_elapsed_fp32_resnet_XNNPACK[:,5]==np.max(job_elapsed_fp32_resnet_XNNPACK[:,5])))

# import sys
# sys.exit()


# plot violin of inference_time
plt.figure(figsize=(16,20), dpi=240)

loop = 0
add_labels_time = []
group_loop = [position_fp32_vgg_XNNPACK, position_fp32_mobilenet_XNNPACK, position_fp32_efficientnet_XNNPACK, position_fp32_inception_XNNPACK, position_fp32_resnet_XNNPACK]
metric_loop = [job_elapsed_fp32_vgg_XNNPACK, job_elapsed_fp32_mobilenet_XNNPACK, job_elapsed_fp32_efficientnet_XNNPACK, job_elapsed_fp32_inception_XNNPACK, job_elapsed_fp32_resnet_XNNPACK]
for a,b in zip(metric_loop, group_loop):
    violin = plt.violinplot(a, positions=b, widths=0.5, vert=True, showmedians=True, showextrema=True, quantiles=[[0.99,0.01]]*len(b))
    for patch in violin['bodies']:
        patch.set_facecolor(colors[loop])
        patch.set_edgecolor(colors[loop])
        patch.set_alpha(0.5)
    for partname in ('cbars','cmins','cmaxes','cmedians','cquantiles'):
        vp = violin[partname]
        vp.set_edgecolor(colors[loop])
        vp.set_linewidth(0.5)
    add_label(add_labels_time, violin, labels[loop])
    loop = loop + 1

plt.legend(*zip(*add_labels_time), loc=1)
plt.grid(alpha=0.5)
plt.xticks(range(17), short_name_fp32_XNNPACK, rotation=75)
plt.yticks(np.arange(0.67, 0.84, 0.005))
plt.xlabel('models')
plt.ylabel("ratio of inference time")
# plt.title('inference time vs. models')
plt.savefig('inference_time-models_violin_XNNPACK.pdf', bbox_inches='tight')

# plot violin of l2 misses
plt.figure(figsize=(16,20), dpi=240)

loop = 0
add_labels_time = []
group_loop = [position_fp32_vgg_XNNPACK, position_fp32_mobilenet_XNNPACK, position_fp32_efficientnet_XNNPACK, position_fp32_inception_XNNPACK, position_fp32_resnet_XNNPACK]
metric_loop = [l2_misses_fp32_vgg_XNNPACK, l2_misses_fp32_mobilenet_XNNPACK, l2_misses_fp32_efficientnet_XNNPACK, l2_misses_fp32_inception_XNNPACK, l2_misses_fp32_resnet_XNNPACK]
for a,b in zip(metric_loop, group_loop):
    violin = plt.violinplot(a, positions=b, widths=0.5, vert=True, showmedians=True, showextrema=True, quantiles=[[0.99,0.01]]*len(b))
    for patch in violin['bodies']:
        patch.set_facecolor(colors[loop])
        patch.set_edgecolor(colors[loop])
        patch.set_alpha(0.5)
    for partname in ('cbars','cmins','cmaxes','cmedians','cquantiles'):
        vp = violin[partname]
        vp.set_edgecolor(colors[loop])
        vp.set_linewidth(0.5)
    add_label(add_labels_time, violin, labels[loop])
    loop = loop + 1

plt.legend(*zip(*add_labels_time), loc=1)
plt.grid(alpha=0.5)
plt.xticks(range(17), short_name_fp32_XNNPACK, rotation=75)
plt.yticks(np.arange(0.7, 2.4, 0.1))
plt.xlabel('models')
plt.ylabel("ratio of l2 misse")
# plt.title('l2 misses vs. models')
plt.savefig('l2_misses-models_violin_XNNPACK.pdf', bbox_inches='tight')

# plot violin of l1 misses
plt.figure(figsize=(16,20), dpi=240)

loop = 0
add_labels_time = []
group_loop = [position_fp32_vgg_XNNPACK, position_fp32_mobilenet_XNNPACK, position_fp32_efficientnet_XNNPACK, position_fp32_inception_XNNPACK, position_fp32_resnet_XNNPACK]
metric_loop = [l1_misses_fp32_vgg_XNNPACK, l1_misses_fp32_mobilenet_XNNPACK, l1_misses_fp32_efficientnet_XNNPACK, l1_misses_fp32_inception_XNNPACK, l1_misses_fp32_resnet_XNNPACK]
for a,b in zip(metric_loop, group_loop):
    violin = plt.violinplot(a, positions=b, widths=0.5, vert=True, showmedians=True, showextrema=True, quantiles=[[0.99,0.01]]*len(b))
    for patch in violin['bodies']:
        patch.set_facecolor(colors[loop])
        patch.set_edgecolor(colors[loop])
        patch.set_alpha(0.5)
    for partname in ('cbars','cmins','cmaxes','cmedians','cquantiles'):
        vp = violin[partname]
        vp.set_edgecolor(colors[loop])
        vp.set_linewidth(0.5)
    add_label(add_labels_time, violin, labels[loop])
    loop = loop + 1

plt.legend(*zip(*add_labels_time), loc=4)
plt.grid(alpha=0.5)
plt.xticks(range(17), short_name_fp32_XNNPACK, rotation=75)
plt.yticks(np.arange(0.55, 1.6, 0.05))
plt.xlabel('models')
plt.ylabel("ratio of l1 misses")
# plt.title('l1 misses vs. models')
plt.savefig('l1_misses-models_violin_XNNPACK.pdf', bbox_inches='tight')

# plot violin of instructions_retied
plt.figure(figsize=(16,20), dpi=240)

loop = 0
add_labels_time = []
group_loop = [position_fp32_vgg_XNNPACK, position_fp32_mobilenet_XNNPACK, position_fp32_efficientnet_XNNPACK, position_fp32_inception_XNNPACK, position_fp32_resnet_XNNPACK]
metric_loop = [instructions_retired_fp32_vgg_XNNPACK, instructions_retired_fp32_mobilenet_XNNPACK, instructions_retired_fp32_efficientnet_XNNPACK, instructions_retired_fp32_inception_XNNPACK, instructions_retired_fp32_resnet_XNNPACK]
for a,b in zip(metric_loop, group_loop):
    violin = plt.violinplot(a, positions=b, widths=0.5, vert=True, showmedians=True, showextrema=True, quantiles=[[0.99,0.01]]*len(b))
    for patch in violin['bodies']:
        patch.set_facecolor(colors[loop])
        patch.set_edgecolor(colors[loop])
        patch.set_alpha(0.5)
    for partname in ('cbars','cmins','cmaxes','cmedians','cquantiles'):
        vp = violin[partname]
        vp.set_edgecolor(colors[loop])
        vp.set_linewidth(0.5)
    add_label(add_labels_time, violin, labels[loop])
    loop = loop + 1

plt.legend(*zip(*add_labels_time), loc=4)
plt.grid(alpha=0.5)
plt.xticks(range(17), short_name_fp32_XNNPACK, rotation=75)
plt.yticks(np.arange(0.91, 0.99, 0.005))
plt.xlabel('models')
plt.ylabel("ratio of retired instructions")
# plt.title('retired instructions vs. models')
plt.savefig('instructions_retired-models_violin_XNNPACK.pdf', bbox_inches='tight')