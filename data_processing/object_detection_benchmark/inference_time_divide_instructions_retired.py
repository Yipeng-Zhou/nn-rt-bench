from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

colors = ["purple", "c", "chocolate", "orange", "green", "royalblue", "red"]
labels = ["00000",
          "0000X", 
          "000X*",
          "00X**",
          "0X***",
          "X****",
          "pre-trained"]

# extract the data from the file "extra_benchmarks.csv"
extra_data = pd.read_csv("extra_benchmarks_full.csv")
models = extra_data["model"].values
short_name = extra_data["short_name"].values
mAP = extra_data["mAP(%)"].values
size = extra_data["size(MB)"].values

# sort associations
association = zip(models, short_name, mAP, size)
association_sorted = sorted(association, key=lambda x:x[1]) # sorted by config
association_sorted.insert(0, association_sorted[-1])
del association_sorted[-1]
sorted_results = zip(*association_sorted)
sorted_models, sorted_short_name, sorted_mAP, sorted_size = [list(x) for x in sorted_results]

# extract the data from the folder "benchmarks"
job_elapsed = []
instructions_retired = []
l1_misses = []
l1_miss_ratio = []
l2_misses = []
l2_miss_ratio = []
cpu_clock_count = []

data_folder = "benchmarks/"
for model in sorted_models:
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

target = np.divide(job_elapsed, instructions_retired)

# group by the type of layers inserted
group_0 = []
group_1 = []
group_2 = []
group_3 = []
group_4 = []
group_5 = []
group_pre = []

for x,y in enumerate(sorted_short_name):
        if(y[0]=='0' and y[1]=='0' and y[2]=='0' and y[3]=='0' and y[4]=='0'):
            group_0.append(x)
        elif(y[0]=='0' and y[1]=='0' and y[2]=='0' and y[3]=='0' and y[4]!='0'):
            group_1.append(x)
        elif(y[0]=='0' and y[1]=='0' and y[2]=='0' and y[3]!='0'):
            group_2.append(x)
        elif(y[0]=='0' and y[1]=='0' and y[2]!='0'):
            group_3.append(x)
        elif(y[0]=='0' and y[1]!='0'):
            group_4.append(x)
        elif(y[0]!='0' and y != 'pre-trained'):
            group_5.append(x)
        else:
            group_pre.append(x)

# plot inference_time/instructions_retired vs. models
group_0_target = []
for x in group_0:
    group_0_target.append(target[x])
group_1_target = []
for x in group_1:
    group_1_target.append(target[x])
group_2_target = []
for x in group_2:
    group_2_target.append(target[x])
group_3_target = []
for x in group_3:
    group_3_target.append(target[x])
group_4_target = []
for x in group_4:
    group_4_target.append(target[x])
group_5_target = []
for x in group_5:
    group_5_target.append(target[x])
group_pre_target = []
for x in group_pre:
    group_pre_target.append(target[x])

plt.figure(figsize=(18,6), dpi=120)

plt.scatter(group_0,group_0_target,color=colors[0],label=labels[0])
plt.scatter(group_1,group_1_target,color=colors[1],label=labels[1])
plt.scatter(group_2,group_2_target,color=colors[2],label=labels[2])
plt.scatter(group_3,group_3_target,color=colors[3],label=labels[3])
plt.scatter(group_4,group_4_target,color=colors[4],label=labels[4])
plt.scatter(group_5,group_5_target,color=colors[5],label=labels[5])
plt.scatter(group_pre,group_pre_target,color=colors[6],label=labels[6])

plt.xticks(range(len(sorted_models)), sorted_short_name, rotation=75)
plt.yticks(np.arange(7.0e-10, 8.4e-10, 0.1e-10))
plt.grid(alpha=0.5)
plt.xlabel('models (arranged by config)')
plt.ylabel('inference time / retired instructions')
plt.legend()
plt.savefig('YOLO_inference_time_divide_instructions_retired.pdf', bbox_inches='tight')
