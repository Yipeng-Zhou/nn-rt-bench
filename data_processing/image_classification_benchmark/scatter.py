from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# extract the data from the file "extra_benchmarks.csv"
extra_data = pd.read_csv("extra_benchmarks_fp32_vgg.csv")
models_fp32_vgg = extra_data["model"].values
size_fp32_vgg = extra_data["size(MB)"].values
short_name_fp32_vgg = extra_data["short_name"].values

extra_data = pd.read_csv("extra_benchmarks_fp32_mobilenet.csv")
models_fp32_mobilenet= extra_data["model"].values
size_fp32_mobilenet = extra_data["size(MB)"].values
short_name_fp32_mobilenet = extra_data["short_name"].values

extra_data = pd.read_csv("extra_benchmarks_fp32_efficientnet.csv")
models_fp32_efficientnet= extra_data["model"].values
size_fp32_efficientnet = extra_data["size(MB)"].values
short_name_fp32_efficientnet = extra_data["short_name"].values

extra_data = pd.read_csv("extra_benchmarks_fp32_inception.csv")
models_fp32_inception= extra_data["model"].values
size_fp32_inception = extra_data["size(MB)"].values
short_name_fp32_inception = extra_data["short_name"].values

extra_data = pd.read_csv("extra_benchmarks_fp32_resnet.csv")
models_fp32_resnet= extra_data["model"].values
size_fp32_resnet = extra_data["size(MB)"].values
short_name_fp32_resnet = extra_data["short_name"].values

extra_data = pd.read_csv("extra_benchmarks_uint8_mobilenet.csv")
models_uint8_mobilenet= extra_data["model"].values
size_uint8_mobilenet = extra_data["size(MB)"].values
short_name_uint8_mobilenet = extra_data["short_name"].values

extra_data = pd.read_csv("extra_benchmarks_uint8_efficientnet.csv")
models_uint8_efficientnet= extra_data["model"].values
size_uint8_efficientnet = extra_data["size(MB)"].values
short_name_uint8_efficientnet = extra_data["short_name"].values

extra_data = pd.read_csv("extra_benchmarks_uint8_inception.csv")
models_uint8_inception= extra_data["model"].values
size_uint8_inception = extra_data["size(MB)"].values
short_name_uint8_inception = extra_data["short_name"].values

# extract the data from the folder "benchmarks"
data_folder = "benchmarks/"

## fp32 mobilenet
job_elapsed_fp32_mobilenet = []
instructions_retired_fp32_mobilenet = []
l1_misses_fp32_mobilenet = []
l1_miss_ratio_fp32_mobilenet = []
l2_misses_fp32_mobilenet = []
l2_miss_ratio_fp32_mobilenet = []
cpu_clock_count_fp32_mobilenet = []

for model in models_fp32_mobilenet:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_fp32_mobilenet += [np.mean(data["job_elapsed(seconds)"].values)]
    instructions_retired_fp32_mobilenet += [np.mean(data["instructions_retired"].values)]
    l1_misses_fp32_mobilenet += [np.mean(data["job_l1_misses"].values)]
    l1_miss_ratio_fp32_mobilenet += [np.mean(data["job_l1_miss_ratio(%%)"].values)]
    l2_misses_fp32_mobilenet += [np.mean(data["job_l2_misses"].values)]
    l2_miss_ratio_fp32_mobilenet += [np.mean(data["job_l2_miss_ratio(%%)"].values)]
    cpu_clock_count_fp32_mobilenet += [np.mean(data["cpu_clock_count"].values)]

bandwidth_fp32_mobilenet = np.divide(l2_misses_fp32_mobilenet, job_elapsed_fp32_mobilenet)
instructions_per_second_fp32_mobilenet = np.divide(instructions_retired_fp32_mobilenet, job_elapsed_fp32_mobilenet)
instructions_per_clock_fp32_mobilenet = np.divide(instructions_retired_fp32_mobilenet, cpu_clock_count_fp32_mobilenet)

flag_sum_fp32_mobilenet = []

for model in models_fp32_mobilenet:
    data = pd.read_csv(data_folder+model+"/"+"prediction_output.csv")
    flag_sum_fp32_mobilenet += [np.sum(data["prediction_flag"].values)]

examples_fp32_mobilenet = np.zeros(np.array(flag_sum_fp32_mobilenet).shape) + 10
top_1_accuracy_fp32_mobilenet = np.divide(flag_sum_fp32_mobilenet, examples_fp32_mobilenet)

## fp32 vgg
job_elapsed_fp32_vgg = []
instructions_retired_fp32_vgg = []
l1_misses_fp32_vgg = []
l1_miss_ratio_fp32_vgg = []
l2_misses_fp32_vgg = []
l2_miss_ratio_fp32_vgg = []
cpu_clock_count_fp32_vgg = []

for model in models_fp32_vgg:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_fp32_vgg += [np.mean(data["job_elapsed(seconds)"].values)]
    instructions_retired_fp32_vgg += [np.mean(data["instructions_retired"].values)]
    l1_misses_fp32_vgg += [np.mean(data["job_l1_misses"].values)]
    l1_miss_ratio_fp32_vgg += [np.mean(data["job_l1_miss_ratio(%%)"].values)]
    l2_misses_fp32_vgg += [np.mean(data["job_l2_misses"].values)]
    l2_miss_ratio_fp32_vgg += [np.mean(data["job_l2_miss_ratio(%%)"].values)]
    cpu_clock_count_fp32_vgg += [np.mean(data["cpu_clock_count"].values)]

bandwidth_fp32_vgg = np.divide(l2_misses_fp32_vgg, job_elapsed_fp32_vgg)
instructions_per_second_fp32_vgg = np.divide(instructions_retired_fp32_vgg, job_elapsed_fp32_vgg)
instructions_per_clock_fp32_vgg = np.divide(instructions_retired_fp32_vgg, cpu_clock_count_fp32_vgg)

flag_sum_fp32_vgg = []

for model in models_fp32_vgg:
    data = pd.read_csv(data_folder+model+"/"+"prediction_output.csv")
    flag_sum_fp32_vgg += [np.sum(data["prediction_flag"].values)]

examples_fp32_vgg = np.zeros(np.array(flag_sum_fp32_vgg).shape) + 10
top_1_accuracy_fp32_vgg = np.divide(flag_sum_fp32_vgg, examples_fp32_vgg)

## fp32 efficientnet
job_elapsed_fp32_efficientnet = []
instructions_retired_fp32_efficientnet = []
l1_misses_fp32_efficientnet = []
l1_miss_ratio_fp32_efficientnet = []
l2_misses_fp32_efficientnet = []
l2_miss_ratio_fp32_efficientnet = []
cpu_clock_count_fp32_efficientnet = []

for model in models_fp32_efficientnet:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_fp32_efficientnet += [np.mean(data["job_elapsed(seconds)"].values)]
    instructions_retired_fp32_efficientnet += [np.mean(data["instructions_retired"].values)]
    l1_misses_fp32_efficientnet += [np.mean(data["job_l1_misses"].values)]
    l1_miss_ratio_fp32_efficientnet += [np.mean(data["job_l1_miss_ratio(%%)"].values)]
    l2_misses_fp32_efficientnet += [np.mean(data["job_l2_misses"].values)]
    l2_miss_ratio_fp32_efficientnet += [np.mean(data["job_l2_miss_ratio(%%)"].values)]
    cpu_clock_count_fp32_efficientnet += [np.mean(data["cpu_clock_count"].values)]

bandwidth_fp32_efficientnet = np.divide(l2_misses_fp32_efficientnet, job_elapsed_fp32_efficientnet)
instructions_per_second_fp32_efficientnet = np.divide(instructions_retired_fp32_efficientnet, job_elapsed_fp32_efficientnet)
instructions_per_clock_fp32_efficientnet = np.divide(instructions_retired_fp32_efficientnet, cpu_clock_count_fp32_efficientnet)

flag_sum_fp32_efficientnet = []

for model in models_fp32_efficientnet:
    data = pd.read_csv(data_folder+model+"/"+"prediction_output.csv")
    flag_sum_fp32_efficientnet += [np.sum(data["prediction_flag"].values)]

examples_fp32_efficientnet = np.zeros(np.array(flag_sum_fp32_efficientnet).shape) + 10
top_1_accuracy_fp32_efficientnet = np.divide(flag_sum_fp32_efficientnet, examples_fp32_efficientnet)

## fp32 inception
job_elapsed_fp32_inception = []
instructions_retired_fp32_inception = []
l1_misses_fp32_inception = []
l1_miss_ratio_fp32_inception = []
l2_misses_fp32_inception = []
l2_miss_ratio_fp32_inception = []
cpu_clock_count_fp32_inception = []

for model in models_fp32_inception:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_fp32_inception += [np.mean(data["job_elapsed(seconds)"].values)]
    instructions_retired_fp32_inception += [np.mean(data["instructions_retired"].values)]
    l1_misses_fp32_inception += [np.mean(data["job_l1_misses"].values)]
    l1_miss_ratio_fp32_inception += [np.mean(data["job_l1_miss_ratio(%%)"].values)]
    l2_misses_fp32_inception += [np.mean(data["job_l2_misses"].values)]
    l2_miss_ratio_fp32_inception += [np.mean(data["job_l2_miss_ratio(%%)"].values)]
    cpu_clock_count_fp32_inception += [np.mean(data["cpu_clock_count"].values)]

bandwidth_fp32_inception = np.divide(l2_misses_fp32_inception, job_elapsed_fp32_inception)
instructions_per_second_fp32_inception = np.divide(instructions_retired_fp32_inception, job_elapsed_fp32_inception)
instructions_per_clock_fp32_inception = np.divide(instructions_retired_fp32_inception, cpu_clock_count_fp32_inception)

flag_sum_fp32_inception = []

for model in models_fp32_inception:
    data = pd.read_csv(data_folder+model+"/"+"prediction_output.csv")
    flag_sum_fp32_inception += [np.sum(data["prediction_flag"].values)]

examples_fp32_inception = np.zeros(np.array(flag_sum_fp32_inception).shape) + 10
top_1_accuracy_fp32_inception = np.divide(flag_sum_fp32_inception, examples_fp32_inception)

## fp32 resnet
job_elapsed_fp32_resnet = []
instructions_retired_fp32_resnet = []
l1_misses_fp32_resnet = []
l1_miss_ratio_fp32_resnet = []
l2_misses_fp32_resnet = []
l2_miss_ratio_fp32_resnet = []
cpu_clock_count_fp32_resnet = []

for model in models_fp32_resnet:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_fp32_resnet += [np.mean(data["job_elapsed(seconds)"].values)]
    instructions_retired_fp32_resnet += [np.mean(data["instructions_retired"].values)]
    l1_misses_fp32_resnet += [np.mean(data["job_l1_misses"].values)]
    l1_miss_ratio_fp32_resnet += [np.mean(data["job_l1_miss_ratio(%%)"].values)]
    l2_misses_fp32_resnet += [np.mean(data["job_l2_misses"].values)]
    l2_miss_ratio_fp32_resnet += [np.mean(data["job_l2_miss_ratio(%%)"].values)]
    cpu_clock_count_fp32_resnet += [np.mean(data["cpu_clock_count"].values)]

bandwidth_fp32_resnet = np.divide(l2_misses_fp32_resnet, job_elapsed_fp32_resnet)
instructions_per_second_fp32_resnet = np.divide(instructions_retired_fp32_resnet, job_elapsed_fp32_resnet)
instructions_per_clock_fp32_resnet = np.divide(instructions_retired_fp32_resnet, cpu_clock_count_fp32_resnet)

flag_sum_fp32_resnet = []

for model in models_fp32_resnet:
    data = pd.read_csv(data_folder+model+"/"+"prediction_output.csv")
    flag_sum_fp32_resnet += [np.sum(data["prediction_flag"].values)]

examples_fp32_resnet = np.zeros(np.array(flag_sum_fp32_resnet).shape) + 10
top_1_accuracy_fp32_resnet = np.divide(flag_sum_fp32_resnet, examples_fp32_resnet)

## uint8 mobilenet
job_elapsed_uint8_mobilenet = []
instructions_retired_uint8_mobilenet = []
l1_misses_uint8_mobilenet = []
l1_miss_ratio_uint8_mobilenet = []
l2_misses_uint8_mobilenet = []
l2_miss_ratio_uint8_mobilenet = []
cpu_clock_count_uint8_mobilenet = []

for model in models_uint8_mobilenet:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_uint8_mobilenet += [np.mean(data["job_elapsed(seconds)"].values)]
    instructions_retired_uint8_mobilenet += [np.mean(data["instructions_retired"].values)]
    l1_misses_uint8_mobilenet += [np.mean(data["job_l1_misses"].values)]
    l1_miss_ratio_uint8_mobilenet += [np.mean(data["job_l1_miss_ratio(%%)"].values)]
    l2_misses_uint8_mobilenet += [np.mean(data["job_l2_misses"].values)]
    l2_miss_ratio_uint8_mobilenet += [np.mean(data["job_l2_miss_ratio(%%)"].values)]
    cpu_clock_count_uint8_mobilenet += [np.mean(data["cpu_clock_count"].values)]

bandwidth_uint8_mobilenet = np.divide(l2_misses_uint8_mobilenet, job_elapsed_uint8_mobilenet)
instructions_per_second_uint8_mobilenet = np.divide(instructions_retired_uint8_mobilenet, job_elapsed_uint8_mobilenet)
instructions_per_clock_uint8_mobilenet = np.divide(instructions_retired_uint8_mobilenet, cpu_clock_count_uint8_mobilenet)

flag_sum_uint8_mobilenet = []

for model in models_uint8_mobilenet:
    data = pd.read_csv(data_folder+model+"/"+"prediction_output.csv")
    flag_sum_uint8_mobilenet += [np.sum(data["prediction_flag"].values)]

examples_uint8_mobilenet = np.zeros(np.array(flag_sum_uint8_mobilenet).shape) + 10
top_1_accuracy_uint8_mobilenet = np.divide(flag_sum_uint8_mobilenet, examples_uint8_mobilenet)

## uint8 efficientnet
job_elapsed_uint8_efficientnet = []
instructions_retired_uint8_efficientnet = []
l1_misses_uint8_efficientnet = []
l1_miss_ratio_uint8_efficientnet = []
l2_misses_uint8_efficientnet = []
l2_miss_ratio_uint8_efficientnet = []
cpu_clock_count_uint8_efficientnet = []

for model in models_uint8_efficientnet:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_uint8_efficientnet += [np.mean(data["job_elapsed(seconds)"].values)]
    instructions_retired_uint8_efficientnet += [np.mean(data["instructions_retired"].values)]
    l1_misses_uint8_efficientnet += [np.mean(data["job_l1_misses"].values)]
    l1_miss_ratio_uint8_efficientnet += [np.mean(data["job_l1_miss_ratio(%%)"].values)]
    l2_misses_uint8_efficientnet += [np.mean(data["job_l2_misses"].values)]
    l2_miss_ratio_uint8_efficientnet += [np.mean(data["job_l2_miss_ratio(%%)"].values)]
    cpu_clock_count_uint8_efficientnet += [np.mean(data["cpu_clock_count"].values)]

bandwidth_uint8_efficientnet = np.divide(l2_misses_uint8_efficientnet, job_elapsed_uint8_efficientnet)
instructions_per_second_uint8_efficientnet = np.divide(instructions_retired_uint8_efficientnet, job_elapsed_uint8_efficientnet)
instructions_per_clock_uint8_efficientnet = np.divide(instructions_retired_uint8_efficientnet, cpu_clock_count_uint8_efficientnet)

flag_sum_uint8_efficientnet = []

for model in models_uint8_efficientnet:
    data = pd.read_csv(data_folder+model+"/"+"prediction_output.csv")
    flag_sum_uint8_efficientnet += [np.sum(data["prediction_flag"].values)]

examples_uint8_efficientnet = np.zeros(np.array(flag_sum_uint8_efficientnet).shape) + 10
top_1_accuracy_uint8_efficientnet = np.divide(flag_sum_uint8_efficientnet, examples_uint8_efficientnet)

## uint8 inception
job_elapsed_uint8_inception = []
instructions_retired_uint8_inception = []
l1_misses_uint8_inception = []
l1_miss_ratio_uint8_inception = []
l2_misses_uint8_inception = []
l2_miss_ratio_uint8_inception = []
cpu_clock_count_uint8_inception = []

for model in models_uint8_inception:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed_uint8_inception += [np.mean(data["job_elapsed(seconds)"].values)]
    instructions_retired_uint8_inception += [np.mean(data["instructions_retired"].values)]
    l1_misses_uint8_inception += [np.mean(data["job_l1_misses"].values)]
    l1_miss_ratio_uint8_inception += [np.mean(data["job_l1_miss_ratio(%%)"].values)]
    l2_misses_uint8_inception += [np.mean(data["job_l2_misses"].values)]
    l2_miss_ratio_uint8_inception += [np.mean(data["job_l2_miss_ratio(%%)"].values)]
    cpu_clock_count_uint8_inception += [np.mean(data["cpu_clock_count"].values)]

bandwidth_uint8_inception = np.divide(l2_misses_uint8_inception, job_elapsed_uint8_inception)
instructions_per_second_uint8_inception = np.divide(instructions_retired_uint8_inception, job_elapsed_uint8_inception)
instructions_per_clock_uint8_inception = np.divide(instructions_retired_uint8_inception, cpu_clock_count_uint8_inception)

flag_sum_uint8_inception = []

for model in models_uint8_inception:
    data = pd.read_csv(data_folder+model+"/"+"prediction_output.csv")
    flag_sum_uint8_inception += [np.sum(data["prediction_flag"].values)]

examples_uint8_inception = np.zeros(np.array(flag_sum_uint8_inception).shape) + 10
top_1_accuracy_uint8_inception = np.divide(flag_sum_uint8_inception, examples_uint8_inception)

# plot inference time vs. top-1 accuracy 
plt.figure(figsize=(18,14), dpi=120)

plt.scatter(top_1_accuracy_fp32_vgg,job_elapsed_fp32_vgg,marker='^',color='lightgreen',label='VGG family with full precision (fp32)')
plt.scatter(top_1_accuracy_fp32_mobilenet,job_elapsed_fp32_mobilenet,marker='^',color='cornflowerblue',label='MobileNet family with full precision (fp32)')
plt.scatter(top_1_accuracy_fp32_efficientnet,job_elapsed_fp32_efficientnet,marker='^',color='mediumorchid',label='EfficientNet family with full precision (fp32)')
plt.scatter(top_1_accuracy_fp32_inception,job_elapsed_fp32_inception,marker='^',color='orange',label='Inception family with full precision (fp32)')
plt.scatter(top_1_accuracy_fp32_resnet,job_elapsed_fp32_resnet,marker='^',color='hotpink',label='ResNet family with full precision (fp32)')
plt.scatter(top_1_accuracy_uint8_mobilenet,job_elapsed_uint8_mobilenet,marker='s',color='cornflowerblue',label='MobileNet family after quantization (uint8)')
plt.scatter(top_1_accuracy_uint8_efficientnet,job_elapsed_uint8_efficientnet,marker='s',color='mediumorchid',label='EfficientNet after quantization (uint8)')
plt.scatter(top_1_accuracy_uint8_inception,job_elapsed_uint8_inception,marker='s',color='orange',label='Inception family after quantization (uint8)')
plt.xlabel('top-1 accuracy (%)')
plt.ylabel('inference time (s)')
plt.xticks(np.arange(65, 79, 0.5))
plt.yticks(np.arange(0, 8, 0.2))
plt.grid(alpha=0.2)
plt.legend(prop=dict(size=14))
# plt.title('inference time w.r.t top-1 accuracy')

plt.annotate(text=short_name_fp32_vgg[0],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_vgg[0],job_elapsed_fp32_vgg[0]+0.02],xytext=[top_1_accuracy_fp32_vgg[0]+0.4,job_elapsed_fp32_vgg[0]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_vgg[1],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_vgg[1],job_elapsed_fp32_vgg[1]-0.03],xytext=[top_1_accuracy_fp32_vgg[1]+0.4,job_elapsed_fp32_vgg[1]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_mobilenet[0],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_mobilenet[0],job_elapsed_fp32_mobilenet[0]+0.02],xytext=[top_1_accuracy_fp32_mobilenet[0]+0.4,job_elapsed_fp32_mobilenet[0]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_mobilenet[1],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_mobilenet[1],job_elapsed_fp32_mobilenet[1]-0.03],xytext=[top_1_accuracy_fp32_mobilenet[1]-1.4,job_elapsed_fp32_mobilenet[1]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_mobilenet[2],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_mobilenet[2],job_elapsed_fp32_mobilenet[2]-0.03],xytext=[top_1_accuracy_fp32_mobilenet[2]+0.4,job_elapsed_fp32_mobilenet[2]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_efficientnet[0],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_efficientnet[0],job_elapsed_fp32_efficientnet[0]-0.03],xytext=[top_1_accuracy_fp32_efficientnet[0]+0.4,job_elapsed_fp32_efficientnet[0]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_inception[0],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_inception[0],job_elapsed_fp32_inception[0]+0.02],xytext=[top_1_accuracy_fp32_inception[0]-1.4,job_elapsed_fp32_inception[0]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_inception[1],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_inception[1],job_elapsed_fp32_inception[1]+0.02],xytext=[top_1_accuracy_fp32_inception[1]+0.4,job_elapsed_fp32_inception[1]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_inception[2],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_inception[2],job_elapsed_fp32_inception[2]+0.02],xytext=[top_1_accuracy_fp32_inception[2]+0.4,job_elapsed_fp32_inception[2]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_inception[3],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_inception[3],job_elapsed_fp32_inception[3]+0.02],xytext=[top_1_accuracy_fp32_inception[3]-1.4,job_elapsed_fp32_inception[3]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_inception[4],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_inception[4],job_elapsed_fp32_inception[4]-0.03],xytext=[top_1_accuracy_fp32_inception[4]-2.0,job_elapsed_fp32_inception[4]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_resnet[0],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_resnet[0],job_elapsed_fp32_resnet[0]+0.02],xytext=[top_1_accuracy_fp32_resnet[0]-1.4,job_elapsed_fp32_resnet[0]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_resnet[1],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_resnet[1],job_elapsed_fp32_resnet[1]+0.02],xytext=[top_1_accuracy_fp32_resnet[1]-1.5,job_elapsed_fp32_resnet[1]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_resnet[2],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_resnet[2],job_elapsed_fp32_resnet[2]+0.02],xytext=[top_1_accuracy_fp32_resnet[2]+0.4,job_elapsed_fp32_resnet[2]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_resnet[3],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_resnet[3],job_elapsed_fp32_resnet[3]+0.02],xytext=[top_1_accuracy_fp32_resnet[3]+0.4,job_elapsed_fp32_resnet[3]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_resnet[4],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_resnet[4],job_elapsed_fp32_resnet[4]-0.03],xytext=[top_1_accuracy_fp32_resnet[4]+0.4,job_elapsed_fp32_resnet[4]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_resnet[5],fontsize=10,fontweight=600,xy=[top_1_accuracy_fp32_resnet[5],job_elapsed_fp32_resnet[5]-0.03],xytext=[top_1_accuracy_fp32_resnet[5]-1.5,job_elapsed_fp32_resnet[5]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))

plt.annotate(text=short_name_uint8_mobilenet[0],fontsize=10,fontweight=600,xy=[top_1_accuracy_uint8_mobilenet[0],job_elapsed_uint8_mobilenet[0]-0.03],xytext=[top_1_accuracy_uint8_mobilenet[0]-1.6,job_elapsed_uint8_mobilenet[0]-0.25],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_mobilenet[1],fontsize=10,fontweight=600,xy=[top_1_accuracy_uint8_mobilenet[1],job_elapsed_uint8_mobilenet[1]+0.02],xytext=[top_1_accuracy_uint8_mobilenet[1]-1.6,job_elapsed_uint8_mobilenet[1]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_mobilenet[2],fontsize=10,fontweight=600,xy=[top_1_accuracy_uint8_mobilenet[2],job_elapsed_uint8_mobilenet[2]+0.02],xytext=[top_1_accuracy_uint8_mobilenet[2]-1.6,job_elapsed_uint8_mobilenet[2]+0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_efficientnet[0],fontsize=10,fontweight=600,xy=[top_1_accuracy_uint8_efficientnet[0],job_elapsed_uint8_efficientnet[0]-0.03],xytext=[top_1_accuracy_uint8_efficientnet[0]-1.8,job_elapsed_uint8_efficientnet[0]-0.25],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_inception[0],fontsize=10,fontweight=600,xy=[top_1_accuracy_uint8_inception[0],job_elapsed_uint8_inception[0]+0.02],xytext=[top_1_accuracy_uint8_inception[0]-1.6,job_elapsed_uint8_inception[0]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_inception[1],fontsize=10,fontweight=600,xy=[top_1_accuracy_uint8_inception[1],job_elapsed_uint8_inception[1]+0.02],xytext=[top_1_accuracy_uint8_inception[1]+0.3,job_elapsed_uint8_inception[1]+0.15],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_inception[2],fontsize=10,fontweight=600,xy=[top_1_accuracy_uint8_inception[2],job_elapsed_uint8_inception[2]+0.02],xytext=[top_1_accuracy_uint8_inception[2]-1.6,job_elapsed_uint8_inception[2]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_inception[3],fontsize=10,fontweight=600,xy=[top_1_accuracy_uint8_inception[3],job_elapsed_uint8_inception[3]+0.02],xytext=[top_1_accuracy_uint8_inception[3]-1.6,job_elapsed_uint8_inception[3]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))

plt.savefig('inference_time-top_1_accuracy.pdf', bbox_inches='tight')

# plot l2 misses vs. inference time
plt.figure(figsize=(18,14), dpi=120)

plt.axes(xscale = "log")
plt.scatter(l2_misses_fp32_vgg,job_elapsed_fp32_vgg,marker='^',color='lightgreen',label='VGG family with full precision (fp32)')
plt.scatter(l2_misses_fp32_mobilenet,job_elapsed_fp32_mobilenet,marker='^',color='cornflowerblue',label='MobileNet family with full precision (fp32)')
plt.scatter(l2_misses_fp32_efficientnet,job_elapsed_fp32_efficientnet,marker='^',color='mediumorchid',label='EfficientNet family with full precision (fp32)')
plt.scatter(l2_misses_fp32_inception,job_elapsed_fp32_inception,marker='^',color='orange',label='Inception family with full precision (fp32)')
plt.scatter(l2_misses_fp32_resnet,job_elapsed_fp32_resnet,marker='^',color='hotpink',label='ResNet family with full precision (fp32)')
plt.scatter(l2_misses_uint8_mobilenet,job_elapsed_uint8_mobilenet,marker='s',color='cornflowerblue',label='MobileNet family after quantization (uint8)')
plt.scatter(l2_misses_uint8_efficientnet,job_elapsed_uint8_efficientnet,marker='s',color='mediumorchid',label='EfficientNet after quantization (uint8)')
plt.scatter(l2_misses_uint8_inception,job_elapsed_uint8_inception,marker='s',color='orange',label='Inception family after quantization (uint8)')
plt.xlabel('l2 misses')
plt.ylabel('inference time (s)')
plt.xlim(0.4e5, 3.0e8)
plt.yticks(np.arange(0, 8, 0.5))
plt.grid(which='both',alpha=0.2)
legend_1 = plt.legend(prop=dict(size=14))
# plt.title('inference time w.r.t l2 misses')

plt.annotate(text=short_name_fp32_vgg[0],fontsize=10,fontweight=600,xy=[l2_misses_fp32_vgg[0],job_elapsed_fp32_vgg[0]+0.02],xytext=[l2_misses_fp32_vgg[0]+0.3e8,job_elapsed_fp32_vgg[0]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_vgg[1],fontsize=10,fontweight=600,xy=[l2_misses_fp32_vgg[1],job_elapsed_fp32_vgg[1]-0.03],xytext=[l2_misses_fp32_vgg[1]+0.4e8,job_elapsed_fp32_vgg[1]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_mobilenet[0],fontsize=10,fontweight=600,xy=[l2_misses_fp32_mobilenet[0],job_elapsed_fp32_mobilenet[0]+0.02],xytext=[l2_misses_fp32_mobilenet[0]-0.7e6,job_elapsed_fp32_mobilenet[0]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_mobilenet[1],fontsize=10,fontweight=600,xy=[l2_misses_fp32_mobilenet[1],job_elapsed_fp32_mobilenet[1]-0.03],xytext=[l2_misses_fp32_mobilenet[1]+0.3e6,job_elapsed_fp32_mobilenet[1]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_mobilenet[2],fontsize=10,fontweight=600,xy=[l2_misses_fp32_mobilenet[2],job_elapsed_fp32_mobilenet[2]-0.03],xytext=[l2_misses_fp32_mobilenet[2]-0.6e6,job_elapsed_fp32_mobilenet[2]-0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_efficientnet[0],fontsize=10,fontweight=600,xy=[l2_misses_fp32_efficientnet[0],job_elapsed_fp32_efficientnet[0]+0.02],xytext=[l2_misses_fp32_efficientnet[0]+0.3e6,job_elapsed_fp32_efficientnet[0]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_inception[0],fontsize=10,fontweight=600,xy=[l2_misses_fp32_inception[0],job_elapsed_fp32_inception[0]+0.02],xytext=[l2_misses_fp32_inception[0]-2.5e6,job_elapsed_fp32_inception[0]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_inception[1],fontsize=10,fontweight=600,xy=[l2_misses_fp32_inception[1],job_elapsed_fp32_inception[1]-0.03],xytext=[l2_misses_fp32_inception[1]+1.5e6,job_elapsed_fp32_inception[1]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_inception[2],fontsize=10,fontweight=600,xy=[l2_misses_fp32_inception[2],job_elapsed_fp32_inception[2]-0.03],xytext=[l2_misses_fp32_inception[2]-1.0e7,job_elapsed_fp32_inception[2]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_inception[3],fontsize=10,fontweight=600,xy=[l2_misses_fp32_inception[3],job_elapsed_fp32_inception[3]+0.02],xytext=[l2_misses_fp32_inception[3]+1.5e7,job_elapsed_fp32_inception[3]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_inception[4],fontsize=10,fontweight=600,xy=[l2_misses_fp32_inception[4],job_elapsed_fp32_inception[4]+0.02],xytext=[l2_misses_fp32_inception[4]-2.2e7,job_elapsed_fp32_inception[4]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_resnet[0],fontsize=10,fontweight=600,xy=[l2_misses_fp32_resnet[0],job_elapsed_fp32_resnet[0]-0.03],xytext=[l2_misses_fp32_resnet[0]-0.8e7,job_elapsed_fp32_resnet[0]-0.25],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_resnet[1],fontsize=10,fontweight=600,xy=[l2_misses_fp32_resnet[1],job_elapsed_fp32_resnet[1]-0.03],xytext=[l2_misses_fp32_resnet[1]-2.1e7,job_elapsed_fp32_resnet[1]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_resnet[2],fontsize=10,fontweight=600,xy=[l2_misses_fp32_resnet[2],job_elapsed_fp32_resnet[2]-0.03],xytext=[l2_misses_fp32_resnet[2]-3.2e7,job_elapsed_fp32_resnet[2]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_resnet[3],fontsize=10,fontweight=600,xy=[l2_misses_fp32_resnet[3],job_elapsed_fp32_resnet[3]+0.02],xytext=[l2_misses_fp32_resnet[3]+0.4e7,job_elapsed_fp32_resnet[3]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_resnet[4],fontsize=10,fontweight=600,xy=[l2_misses_fp32_resnet[4],job_elapsed_fp32_resnet[4]+0.02],xytext=[l2_misses_fp32_resnet[4]+1.0e7,job_elapsed_fp32_resnet[4]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_fp32_resnet[5],fontsize=10,fontweight=600,xy=[l2_misses_fp32_resnet[5],job_elapsed_fp32_resnet[5]+0.02],xytext=[l2_misses_fp32_resnet[5]+2.0e7,job_elapsed_fp32_resnet[5]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))

plt.annotate(text=short_name_uint8_mobilenet[0],fontsize=10,fontweight=600,xy=[l2_misses_uint8_mobilenet[0],job_elapsed_uint8_mobilenet[0]+0.02],xytext=[l2_misses_uint8_mobilenet[0]-1.1e5,job_elapsed_uint8_mobilenet[0]+0.15],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_mobilenet[1],fontsize=10,fontweight=600,xy=[l2_misses_uint8_mobilenet[1],job_elapsed_uint8_mobilenet[1]-0.03],xytext=[l2_misses_uint8_mobilenet[1]+0.3e5,job_elapsed_uint8_mobilenet[1]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_mobilenet[2],fontsize=10,fontweight=600,xy=[l2_misses_uint8_mobilenet[2],job_elapsed_uint8_mobilenet[2]-0.03],xytext=[l2_misses_uint8_mobilenet[2]-1.0e5,job_elapsed_uint8_mobilenet[2]-0.3],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_efficientnet[0],fontsize=10,fontweight=600,xy=[l2_misses_uint8_efficientnet[0],job_elapsed_uint8_efficientnet[0]+0.02],xytext=[l2_misses_uint8_efficientnet[0]-1.4e5,job_elapsed_uint8_efficientnet[0]+0.25],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_inception[0],fontsize=10,fontweight=600,xy=[l2_misses_uint8_inception[0],job_elapsed_uint8_inception[0]+0.02],xytext=[l2_misses_uint8_inception[0]-2.2e5,job_elapsed_uint8_inception[0]+0.35],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_inception[1],fontsize=10,fontweight=600,xy=[l2_misses_uint8_inception[1],job_elapsed_uint8_inception[1]+0.02],xytext=[l2_misses_uint8_inception[1]+1.2e5,job_elapsed_uint8_inception[1]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_inception[2],fontsize=10,fontweight=600,xy=[l2_misses_uint8_inception[2],job_elapsed_uint8_inception[2]+0.02],xytext=[l2_misses_uint8_inception[2]+0.5e6,job_elapsed_uint8_inception[2]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))
plt.annotate(text=short_name_uint8_inception[3],fontsize=10,fontweight=600,xy=[l2_misses_uint8_inception[3],job_elapsed_uint8_inception[3]+0.02],xytext=[l2_misses_uint8_inception[3]+1.0e6,job_elapsed_uint8_inception[3]+0.2],
    arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5'))

y = np.arange(0,8,0.01)
x_1 = y * 0.058e7
x_2 = y * 0.082e7
x_3 = y * 0.097e7
x_4 = y * 0.320e7
x_5 = y * 0.460e7
x_6 = y * 0.720e7
x_7 = y * 1.180e7
x_8 = y * 1.790e7
bandwidth_1, = plt.plot(x_1,y,alpha=0.2)
bandwidth_2, = plt.plot(x_2,y,alpha=0.2)
bandwidth_3, = plt.plot(x_3,y,alpha=0.2)
bandwidth_4, = plt.plot(x_4,y,alpha=0.2)
bandwidth_5, = plt.plot(x_5,y,alpha=0.2)
bandwidth_6, = plt.plot(x_6,y,alpha=0.2)
bandwidth_7, = plt.plot(x_7,y,alpha=0.2)
bandwidth_8, = plt.plot(x_8,y,alpha=0.2)
legend_2 = plt.legend([bandwidth_1,bandwidth_2,bandwidth_3,bandwidth_4,bandwidth_5,bandwidth_6,bandwidth_7,bandwidth_8],
            ['bandwidth = 0.035 GB/s','bandwidth = 0.049 GB/s','bandwidth = 0.059 GB/s','bandwidth = 0.191 GB/s','bandwidth = 0.274 GB/s','bandwidth = 0.429 GB/s','bandwidth = 0.703 GB/s','bandwidth = 1.067 GB/s'],
            loc='lower right',prop=dict(size=12))
plt.gca().add_artist(legend_1)

plt.savefig('inference_time-l2_misses.pdf', bbox_inches='tight')
