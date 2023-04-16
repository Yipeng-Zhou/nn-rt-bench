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
association_sorted = sorted(association, key=lambda x:x[1]) # sorted by names
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

# plot size of models vs. models
group_0_size = []
for x in group_0:
    group_0_size.append(sorted_size[x])
group_1_size = []
for x in group_1:
    group_1_size.append(sorted_size[x])
group_2_size = []
for x in group_2:
    group_2_size.append(sorted_size[x])
group_3_size = []
for x in group_3:
    group_3_size.append(sorted_size[x])
group_4_size = []
for x in group_4:
    group_4_size.append(sorted_size[x])
group_5_size = []
for x in group_5:
    group_5_size.append(sorted_size[x])
group_pre_size = []
for x in group_pre:
    group_pre_size.append(sorted_size[x])

plt.figure(figsize=(18,6), dpi=120)

plt.scatter(group_0,group_0_size,color=colors[0],label=labels[0])
plt.scatter(group_1,group_1_size,color=colors[1],label=labels[1])
plt.scatter(group_2,group_2_size,color=colors[2],label=labels[2])
plt.scatter(group_3,group_3_size,color=colors[3],label=labels[3])
plt.scatter(group_4,group_4_size,color=colors[4],label=labels[4])
plt.scatter(group_5,group_5_size,color=colors[5],label=labels[5])
plt.scatter(group_pre,group_pre_size,color=colors[6],label=labels[6])

plt.xticks(range(len(sorted_models)), sorted_short_name, rotation=60)
plt.grid()
plt.xlabel('models')
plt.ylabel('size of models (MB)')
plt.legend()
plt.title('size vs. models')
plt.savefig('size-models.png', bbox_inches='tight')

# plot inference time vs. models
group_0_time = []
for x in group_0:
    group_0_time.append(job_elapsed[x])
group_1_time = []
for x in group_1:
    group_1_time.append(job_elapsed[x])
group_2_time = []
for x in group_2:
    group_2_time.append(job_elapsed[x])
group_3_time = []
for x in group_3:
    group_3_time.append(job_elapsed[x])
group_4_time = []
for x in group_4:
    group_4_time.append(job_elapsed[x])
group_5_time = []
for x in group_5:
    group_5_time.append(job_elapsed[x])
group_pre_time = []
for x in group_pre:
    group_pre_time.append(job_elapsed[x])

plt.figure(figsize=(18,6), dpi=120)

plt.scatter(group_0,group_0_time,color=colors[0],label=labels[0])
plt.scatter(group_1,group_1_time,color=colors[1],label=labels[1])
plt.scatter(group_2,group_2_time,color=colors[2],label=labels[2])
plt.scatter(group_3,group_3_time,color=colors[3],label=labels[3])
plt.scatter(group_4,group_4_time,color=colors[4],label=labels[4])
plt.scatter(group_5,group_5_time,color=colors[5],label=labels[5])
plt.scatter(group_pre,group_pre_time,color=colors[6],label=labels[6])

plt.xticks(range(len(sorted_models)), sorted_short_name, rotation=60)
plt.grid()
plt.xlabel('models')
plt.ylabel('inference time (s)')
plt.legend()
plt.title('inference time vs. models')
plt.savefig('inference_time-models.png', bbox_inches='tight')

# plot instructions_retired vs. models
group_0_instructions = []
for x in group_0:
    group_0_instructions.append(instructions_retired[x])
group_1_instructions = []
for x in group_1:
    group_1_instructions.append(instructions_retired[x])
group_2_instructions = []
for x in group_2:
    group_2_instructions.append(instructions_retired[x])
group_3_instructions = []
for x in group_3:
    group_3_instructions.append(instructions_retired[x])
group_4_instructions = []
for x in group_4:
    group_4_instructions.append(instructions_retired[x])
group_5_instructions = []
for x in group_5:
    group_5_instructions.append(instructions_retired[x])
group_pre_instructions = []
for x in group_pre:
    group_pre_instructions.append(instructions_retired[x])

plt.figure(figsize=(18,6), dpi=120)

plt.scatter(group_0,group_0_instructions,color=colors[0],label=labels[0])
plt.scatter(group_1,group_1_instructions,color=colors[1],label=labels[1])
plt.scatter(group_2,group_2_instructions,color=colors[2],label=labels[2])
plt.scatter(group_3,group_3_instructions,color=colors[3],label=labels[3])
plt.scatter(group_4,group_4_instructions,color=colors[4],label=labels[4])
plt.scatter(group_5,group_5_instructions,color=colors[5],label=labels[5])
plt.scatter(group_pre,group_pre_instructions,color=colors[6],label=labels[6])

plt.xticks(range(len(sorted_models)), sorted_short_name, rotation=60)
plt.grid()
plt.xlabel('models')
plt.ylabel('retired instructions')
plt.legend()
plt.title('retired instructions vs. models')
plt.savefig('instructions_retired-models.png', bbox_inches='tight')

# plot instructions_per_second vs. models
group_0_instructions_per_second = []
for x in group_0:
    group_0_instructions_per_second.append(instructions_per_second[x])
group_1_instructions_per_second = []
for x in group_1:
    group_1_instructions_per_second.append(instructions_per_second[x])
group_2_instructions_per_second = []
for x in group_2:
    group_2_instructions_per_second.append(instructions_per_second[x])
group_3_instructions_per_second = []
for x in group_3:
    group_3_instructions_per_second.append(instructions_per_second[x])
group_4_instructions_per_second = []
for x in group_4:
    group_4_instructions_per_second.append(instructions_per_second[x])
group_5_instructions_per_second = []
for x in group_5:
    group_5_instructions_per_second.append(instructions_per_second[x])
group_pre_instructions_per_second = []
for x in group_pre:
    group_pre_instructions_per_second.append(instructions_per_second[x])

plt.figure(figsize=(18,6), dpi=120)

plt.scatter(group_0,group_0_instructions_per_second,color=colors[0],label=labels[0])
plt.scatter(group_1,group_1_instructions_per_second,color=colors[1],label=labels[1])
plt.scatter(group_2,group_2_instructions_per_second,color=colors[2],label=labels[2])
plt.scatter(group_3,group_3_instructions_per_second,color=colors[3],label=labels[3])
plt.scatter(group_4,group_4_instructions_per_second,color=colors[4],label=labels[4])
plt.scatter(group_5,group_5_instructions_per_second,color=colors[5],label=labels[5])
plt.scatter(group_pre,group_pre_instructions_per_second,color=colors[6],label=labels[6])

plt.xticks(range(len(sorted_models)), sorted_short_name, rotation=60)
plt.grid()
plt.xlabel('models')
plt.ylabel('instructions per second')
plt.legend()
plt.title('instructions per second vs. models')
plt.savefig('instructions_per_second-models.png', bbox_inches='tight')

# plot instructions_per_clock vs. models
group_0_instructions_per_clock = []
for x in group_0:
    group_0_instructions_per_clock.append(instructions_per_clock[x])
group_1_instructions_per_clock = []
for x in group_1:
    group_1_instructions_per_clock.append(instructions_per_clock[x])
group_2_instructions_per_clock = []
for x in group_2:
    group_2_instructions_per_clock.append(instructions_per_clock[x])
group_3_instructions_per_clock = []
for x in group_3:
    group_3_instructions_per_clock.append(instructions_per_clock[x])
group_4_instructions_per_clock = []
for x in group_4:
    group_4_instructions_per_clock.append(instructions_per_clock[x])
group_5_instructions_per_clock = []
for x in group_5:
    group_5_instructions_per_clock.append(instructions_per_clock[x])
group_pre_instructions_per_clock = []
for x in group_pre:
    group_pre_instructions_per_clock.append(instructions_per_clock[x])

plt.figure(figsize=(18,6), dpi=120)

plt.scatter(group_0,group_0_instructions_per_clock,color=colors[0],label=labels[0])
plt.scatter(group_1,group_1_instructions_per_clock,color=colors[1],label=labels[1])
plt.scatter(group_2,group_2_instructions_per_clock,color=colors[2],label=labels[2])
plt.scatter(group_3,group_3_instructions_per_clock,color=colors[3],label=labels[3])
plt.scatter(group_4,group_4_instructions_per_clock,color=colors[4],label=labels[4])
plt.scatter(group_5,group_5_instructions_per_clock,color=colors[5],label=labels[5])
plt.scatter(group_pre,group_pre_instructions_per_clock,color=colors[6],label=labels[6])

plt.xticks(range(len(sorted_models)), sorted_short_name, rotation=60)
plt.grid()
plt.xlabel('models')
plt.ylabel('instructions per clock')
plt.legend()
plt.title('instructions per clock (IPC) vs. models')
plt.savefig('instructions_per_clock-models.png', bbox_inches='tight')

# plot mAP50 vs. models
group_0_mAP50 = []
for x in group_0:
    group_0_mAP50.append(sorted_mAP[x])
group_1_mAP50 = []
for x in group_1:
    group_1_mAP50.append(sorted_mAP[x])
group_2_mAP50 = []
for x in group_2:
    group_2_mAP50.append(sorted_mAP[x])
group_3_mAP50 = []
for x in group_3:
    group_3_mAP50.append(sorted_mAP[x])
group_4_mAP50 = []
for x in group_4:
    group_4_mAP50.append(sorted_mAP[x])
group_5_mAP50 = []
for x in group_5:
    group_5_mAP50.append(sorted_mAP[x])
group_pre_mAP50 = []
for x in group_pre:
    group_pre_mAP50.append(sorted_mAP[x])

plt.figure(figsize=(18,6), dpi=120)

plt.scatter(group_0,group_0_mAP50,color=colors[0],label=labels[0])
plt.scatter(group_1,group_1_mAP50,color=colors[1],label=labels[1])
plt.scatter(group_2,group_2_mAP50,color=colors[2],label=labels[2])
plt.scatter(group_3,group_3_mAP50,color=colors[3],label=labels[3])
plt.scatter(group_4,group_4_mAP50,color=colors[4],label=labels[4])
plt.scatter(group_5,group_5_mAP50,color=colors[5],label=labels[5])
plt.scatter(group_pre,group_pre_mAP50,color=colors[6],label=labels[6])

plt.xticks(range(len(sorted_models)), sorted_short_name, rotation=60)
plt.grid()
plt.xlabel('models')
plt.ylabel('mAP50 (%)')
plt.legend()
plt.title('mAP50 vs. models')
plt.savefig('mAP50-models.png', bbox_inches='tight')

# plot l1_misses vs. models
group_0_l1_misses = []
for x in group_0:
    group_0_l1_misses.append(l1_misses[x])
group_1_l1_misses = []
for x in group_1:
    group_1_l1_misses.append(l1_misses[x])
group_2_l1_misses = []
for x in group_2:
    group_2_l1_misses.append(l1_misses[x])
group_3_l1_misses = []
for x in group_3:
    group_3_l1_misses.append(l1_misses[x])
group_4_l1_misses = []
for x in group_4:
    group_4_l1_misses.append(l1_misses[x])
group_5_l1_misses = []
for x in group_5:
    group_5_l1_misses.append(l1_misses[x])
group_pre_l1_misses = []
for x in group_pre:
    group_pre_l1_misses.append(l1_misses[x])

plt.figure(figsize=(18,6), dpi=120)

plt.scatter(group_0,group_0_l1_misses,color=colors[0],label=labels[0])
plt.scatter(group_1,group_1_l1_misses,color=colors[1],label=labels[1])
plt.scatter(group_2,group_2_l1_misses,color=colors[2],label=labels[2])
plt.scatter(group_3,group_3_l1_misses,color=colors[3],label=labels[3])
plt.scatter(group_4,group_4_l1_misses,color=colors[4],label=labels[4])
plt.scatter(group_5,group_5_l1_misses,color=colors[5],label=labels[5])
plt.scatter(group_pre,group_pre_l1_misses,color=colors[6],label=labels[6])

plt.xticks(range(len(sorted_models)), sorted_short_name, rotation=60)
plt.grid()
plt.xlabel('models')
plt.ylabel('l1 misses')
plt.legend()
plt.title('l1 misses vs. models')
plt.savefig('l1_misses-models.png', bbox_inches='tight')

# plot l1_miss_ratio vs. models
group_0_l1_miss_ratio = []
for x in group_0:
    group_0_l1_miss_ratio.append(l1_miss_ratio[x])
group_1_l1_miss_ratio = []
for x in group_1:
    group_1_l1_miss_ratio.append(l1_miss_ratio[x])
group_2_l1_miss_ratio = []
for x in group_2:
    group_2_l1_miss_ratio.append(l1_miss_ratio[x])
group_3_l1_miss_ratio = []
for x in group_3:
    group_3_l1_miss_ratio.append(l1_miss_ratio[x])
group_4_l1_miss_ratio = []
for x in group_4:
    group_4_l1_miss_ratio.append(l1_miss_ratio[x])
group_5_l1_miss_ratio = []
for x in group_5:
    group_5_l1_miss_ratio.append(l1_miss_ratio[x])
group_pre_l1_miss_ratio = []
for x in group_pre:
    group_pre_l1_miss_ratio.append(l1_miss_ratio[x])

plt.figure(figsize=(18,6), dpi=120)

plt.scatter(group_0,group_0_l1_miss_ratio,color=colors[0],label=labels[0])
plt.scatter(group_1,group_1_l1_miss_ratio,color=colors[1],label=labels[1])
plt.scatter(group_2,group_2_l1_miss_ratio,color=colors[2],label=labels[2])
plt.scatter(group_3,group_3_l1_miss_ratio,color=colors[3],label=labels[3])
plt.scatter(group_4,group_4_l1_miss_ratio,color=colors[4],label=labels[4])
plt.scatter(group_5,group_5_l1_miss_ratio,color=colors[5],label=labels[5])
plt.scatter(group_pre,group_pre_l1_miss_ratio,color=colors[6],label=labels[6])

plt.xticks(range(len(sorted_models)), sorted_short_name, rotation=60)
plt.grid()
plt.xlabel('models')
plt.ylabel('l1 miss ratio')
plt.legend()
plt.title('l1 miss ratio vs. models')
plt.savefig('l1_miss_ratio-models.png', bbox_inches='tight')

# plot l2_misses vs. models
group_0_l2_misses = []
for x in group_0:
    group_0_l2_misses.append(l2_misses[x])
group_1_l2_misses = []
for x in group_1:
    group_1_l2_misses.append(l2_misses[x])
group_2_l2_misses = []
for x in group_2:
    group_2_l2_misses.append(l2_misses[x])
group_3_l2_misses = []
for x in group_3:
    group_3_l2_misses.append(l2_misses[x])
group_4_l2_misses = []
for x in group_4:
    group_4_l2_misses.append(l2_misses[x])
group_5_l2_misses = []
for x in group_5:
    group_5_l2_misses.append(l2_misses[x])
group_pre_l2_misses = []
for x in group_pre:
    group_pre_l2_misses.append(l2_misses[x])

plt.figure(figsize=(18,6), dpi=120)

plt.scatter(group_0,group_0_l2_misses,color=colors[0],label=labels[0])
plt.scatter(group_1,group_1_l2_misses,color=colors[1],label=labels[1])
plt.scatter(group_2,group_2_l2_misses,color=colors[2],label=labels[2])
plt.scatter(group_3,group_3_l2_misses,color=colors[3],label=labels[3])
plt.scatter(group_4,group_4_l2_misses,color=colors[4],label=labels[4])
plt.scatter(group_5,group_5_l2_misses,color=colors[5],label=labels[5])
plt.scatter(group_pre,group_pre_l2_misses,color=colors[6],label=labels[6])

plt.xticks(range(len(sorted_models)), sorted_short_name, rotation=60)
plt.grid()
plt.xlabel('models')
plt.ylabel('l2 misses')
plt.legend()
plt.title('l2 misses vs. models')
plt.savefig('l2_misses-models.png', bbox_inches='tight')

# plot l2_miss_ratio vs. models
group_0_l2_miss_ratio = []
for x in group_0:
    group_0_l2_miss_ratio.append(l2_miss_ratio[x])
group_1_l2_miss_ratio = []
for x in group_1:
    group_1_l2_miss_ratio.append(l2_miss_ratio[x])
group_2_l2_miss_ratio = []
for x in group_2:
    group_2_l2_miss_ratio.append(l2_miss_ratio[x])
group_3_l2_miss_ratio = []
for x in group_3:
    group_3_l2_miss_ratio.append(l2_miss_ratio[x])
group_4_l2_miss_ratio = []
for x in group_4:
    group_4_l2_miss_ratio.append(l2_miss_ratio[x])
group_5_l2_miss_ratio = []
for x in group_5:
    group_5_l2_miss_ratio.append(l2_miss_ratio[x])
group_pre_l2_miss_ratio = []
for x in group_pre:
    group_pre_l2_miss_ratio.append(l2_miss_ratio[x])

plt.figure(figsize=(18,6), dpi=120)

plt.scatter(group_0,group_0_l2_miss_ratio,color=colors[0],label=labels[0])
plt.scatter(group_1,group_1_l2_miss_ratio,color=colors[1],label=labels[1])
plt.scatter(group_2,group_2_l2_miss_ratio,color=colors[2],label=labels[2])
plt.scatter(group_3,group_3_l2_miss_ratio,color=colors[3],label=labels[3])
plt.scatter(group_4,group_4_l2_miss_ratio,color=colors[4],label=labels[4])
plt.scatter(group_5,group_5_l2_miss_ratio,color=colors[5],label=labels[5])
plt.scatter(group_pre,group_pre_l2_miss_ratio,color=colors[6],label=labels[6])

plt.xticks(range(len(sorted_models)), sorted_short_name, rotation=60)
plt.grid()
plt.xlabel('models')
plt.ylabel('l2 miss ratio')
plt.legend()
plt.title('l2 miss ratio vs. models')
plt.savefig('l2_miss_ratio-models.png', bbox_inches='tight')

# plot bandwidth vs. models
group_0_bandwidth = []
for x in group_0:
    group_0_bandwidth.append(bandwidth[x])
group_1_bandwidth = []
for x in group_1:
    group_1_bandwidth.append(bandwidth[x])
group_2_bandwidth = []
for x in group_2:
    group_2_bandwidth.append(bandwidth[x])
group_3_bandwidth = []
for x in group_3:
    group_3_bandwidth.append(bandwidth[x])
group_4_bandwidth = []
for x in group_4:
    group_4_bandwidth.append(bandwidth[x])
group_5_bandwidth = []
for x in group_5:
    group_5_bandwidth.append(bandwidth[x])
group_pre_bandwidth = []
for x in group_pre:
    group_pre_bandwidth.append(bandwidth[x])

plt.figure(figsize=(18,6), dpi=120)

plt.scatter(group_0,group_0_bandwidth,color=colors[0],label=labels[0])
plt.scatter(group_1,group_1_bandwidth,color=colors[1],label=labels[1])
plt.scatter(group_2,group_2_bandwidth,color=colors[2],label=labels[2])
plt.scatter(group_3,group_3_bandwidth,color=colors[3],label=labels[3])
plt.scatter(group_4,group_4_bandwidth,color=colors[4],label=labels[4])
plt.scatter(group_5,group_5_bandwidth,color=colors[5],label=labels[5])
plt.scatter(group_pre,group_pre_bandwidth,color=colors[6],label=labels[6])

plt.xticks(range(len(sorted_models)), sorted_short_name, rotation=60)
plt.grid()
plt.xlabel('models')
plt.ylabel('bandwidth')
plt.legend()
plt.title('bandwidth vs. models')
plt.savefig('bandwidth-models.png', bbox_inches='tight')

# plot l2 misses vs. retired instructions
plt.figure(figsize=(18,6), dpi=120)
plt.scatter(group_0_instructions,group_0_l2_misses,color=colors[0],label=labels[0])
plt.scatter(group_1_instructions,group_1_l2_misses,color=colors[1],label=labels[1])
plt.scatter(group_2_instructions,group_2_l2_misses,color=colors[2],label=labels[2])
plt.scatter(group_3_instructions,group_3_l2_misses,color=colors[3],label=labels[3])
plt.scatter(group_4_instructions,group_4_l2_misses,color=colors[4],label=labels[4])
plt.scatter(group_5_instructions,group_5_l2_misses,color=colors[5],label=labels[5])
plt.scatter(group_pre_instructions,group_pre_l2_misses,color=colors[6],label=labels[6])

plt.grid()
plt.xlabel('retired instructions')
plt.ylabel('l2 misses')
plt.legend()
plt.title('l2 misses vs. retired instructions')
plt.savefig('l2_misses-instructions_retired.png', bbox_inches='tight')



