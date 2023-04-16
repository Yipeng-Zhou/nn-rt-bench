from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

colors = ["red", "purple", "olive", "brown", "orange", "green", "blue"]
labels = ["pre-trained",
          "00000",
          "00001", 
          "00010",
          "00100",
          "01000",
          "10000"]

# extract the data from the file "extra_benchmarks_one_inserted_layer.csv"
extra_data = pd.read_csv("extra_benchmarks_one_layer_insert.csv")
models = extra_data["model"].values
short_name = extra_data["short_name"].values
mAP = extra_data["mAP(%)"].values
size = extra_data["size(MB)"].values

# extract the data from the folder "benchmarks"
job_elapsed = []
instructions_retired = []
l1_misses = []
l1_miss_ratio = []
l2_misses = []
l2_miss_ratio = []
cpu_clock_count = []

data_folder = "benchmarks/"
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

# plot size of models vs. models
size = size / size[1]
plt.figure(figsize=(15,6), dpi=80)
plt.bar(models, size, width=0.5, fc='orange')
plt.xticks(models, short_name)
plt.ylim(0.975,1.075)
plt.xlabel('models')
plt.ylabel('size of models (MB)')
plt.title('size of models vs. models')

for a,b in zip(models, size):
    plt.text(a, b, '%.5f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('size-models.png', bbox_inches='tight')

# plot inference time vs. models
job_elapsed = job_elapsed / job_elapsed[1]
plt.figure(figsize=(15,6), dpi=80)
plt.bar(models, job_elapsed, width=0.5, fc='orange')
plt.xticks(models, short_name)
plt.ylim(0.9,1.3)
plt.xlabel('models')
plt.ylabel('inference time (s)')
plt.title('inference time vs. models')

for a,b in zip(models, job_elapsed):
    plt.text(a, b, '%.5f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('inference_time-models.png', bbox_inches='tight')

# plot instructions_retired vs. models
instructions_retired = instructions_retired / instructions_retired[1]
plt.figure(figsize=(15,6), dpi=80)
plt.bar(models, instructions_retired, width=0.5, fc='orange')
plt.xticks(models, short_name)
plt.ylim(0.9,1.25)
plt.xlabel('models')
plt.ylabel('retired instructions')
plt.title('retired instructions vs. models')

for a,b in zip(models, instructions_retired):
    plt.text(a, b, '%.5f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('instructions_retired-models.png', bbox_inches='tight')

# plot instructions_per_second vs. models
instructions_per_second = instructions_per_second / instructions_per_second[1]
plt.figure(figsize=(15,6), dpi=80)
plt.bar(models, instructions_per_second, width=0.5, fc='orange')
plt.xticks(models, short_name)
plt.ylim(0.94,1.02)
plt.xlabel('models')
plt.ylabel('instructions per second')
plt.title('instructions per second vs. models')

for a,b in zip(models, instructions_per_second):
    plt.text(a, b, '%.5f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('instructions_per_second-models.png', bbox_inches='tight')

# plot instructions_per_clock vs. models
instructions_per_clock = instructions_per_clock / instructions_per_clock[1]
plt.figure(figsize=(15,6), dpi=80)
plt.bar(models, instructions_per_clock, width=0.5, fc='orange')
plt.xticks(models, short_name)
plt.ylim(0.94,1.02)
plt.xlabel('models')
plt.ylabel('instructions per clock')
plt.title('instructions per clock (IPC) vs. models')

for a,b in zip(models, instructions_per_clock):
    plt.text(a, b, '%.5f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('instructions_per_clock-models.png', bbox_inches='tight')

# plot mAP50 vs. models
mAP = mAP / mAP[1]
plt.figure(figsize=(15,6), dpi=80)
plt.bar(models, mAP, width=0.5, fc='orange')
plt.xticks(models, short_name)
plt.ylim(0.8,1.4)
plt.xlabel('models')
plt.ylabel('mAP50 (%)')
plt.title('mAP50 vs. models')

for a,b in zip(models, mAP):
    plt.text(a, b, '%.5f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('mAP50-models.png', bbox_inches='tight')

# plot l1_misses vs. models
l1_misses = l1_misses / l1_misses[1]
plt.figure(figsize=(15,6), dpi=80)
plt.bar(models, l1_misses, width=0.5, fc='orange')
plt.xticks(models, short_name)
plt.ylim(0.95,1.4)
plt.xlabel('models')
plt.ylabel('l1 misses')
plt.title('l1 misses vs. models')

for a,b in zip(models, l1_misses):
    plt.text(a, b, '%.5f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('l1_misses-models.png', bbox_inches='tight')

# plot l1_miss_ratio vs. models
l1_miss_ratio = l1_miss_ratio / l1_miss_ratio[1]
plt.figure(figsize=(15,6), dpi=80)
plt.bar(models, l1_miss_ratio, width=0.5, fc='orange')
plt.xticks(models, short_name)
plt.ylim(0.975,1.15)
plt.xlabel('models')
plt.ylabel('l1 miss ratio')
plt.title('l1 miss ratio vs. models')

for a,b in zip(models, l1_miss_ratio):
    plt.text(a, b, '%.5f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('l1_miss_ratio-models.png', bbox_inches='tight')

# plot l2_misses vs. models
l2_misses = l2_misses / l2_misses[1]
plt.figure(figsize=(15,6), dpi=80)
plt.bar(models, l2_misses, width=0.5, fc='orange')
plt.xticks(models, short_name)
plt.ylim(0.9,1.2)
plt.xlabel('models')
plt.ylabel('l2 misses')
plt.title('l2 misses vs. models')

for a,b in zip(models, l2_misses):
    plt.text(a, b, '%.5f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('l2_misses-models.png', bbox_inches='tight')

# plot l2_miss_ratio vs. models
l2_miss_ratio = l2_miss_ratio / l2_miss_ratio[1]
plt.figure(figsize=(15,6), dpi=80)
plt.bar(models, l2_miss_ratio, width=0.5, fc='orange')
plt.xticks(models, short_name)
plt.ylim(0.85,1.05)
plt.xlabel('models')
plt.ylabel('l2 miss ratio')
plt.title('l2 miss ratio vs. models')

for a,b in zip(models, l2_miss_ratio):
    plt.text(a, b, '%.5f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('l2_miss_ratio-models.png', bbox_inches='tight')

# plot bandwidth vs. models
bandwidth = bandwidth / bandwidth[1]
plt.figure(figsize=(15,6), dpi=80)
plt.bar(models, bandwidth, width=0.5, fc='orange')
plt.xticks(models, short_name)
plt.ylim(0.875,1.05)
plt.xlabel('models')
plt.ylabel('bandwidth')
plt.title('bandwidth vs. models')

for a,b in zip(models, bandwidth):
    plt.text(a, b, '%.5f'%b, ha='center', va='bottom', fontsize=10)

plt.savefig('bandwidth-models.png', bbox_inches='tight')

# plot l2 misses vs. retired instructions
plt.figure(figsize=(18,6), dpi=120)
plt.scatter(instructions_retired[0],l2_misses[0],color=colors[0],label=labels[0])
plt.scatter(instructions_retired[1],l2_misses[1],color=colors[1],label=labels[1])
plt.scatter(instructions_retired[2],l2_misses[2],color=colors[2],label=labels[2])
plt.scatter(instructions_retired[3],l2_misses[3],color=colors[3],label=labels[3])
plt.scatter(instructions_retired[4],l2_misses[4],color=colors[4],label=labels[4])
plt.scatter(instructions_retired[5],l2_misses[5],color=colors[5],label=labels[5])
plt.scatter(instructions_retired[6],l2_misses[6],color=colors[6],label=labels[6])

plt.grid()
plt.xlabel('retired instructions')
plt.ylabel('l2 misses')
plt.legend()
plt.title('l2 misses vs. retired instructions')
plt.savefig('l2_misses-instructions_retired.png', bbox_inches='tight')


