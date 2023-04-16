from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

colors = ["purple", "c", "chocolate", "orange", "green", "royalblue", "red", "lightgray"]
labels = ["00000",
          "0000X", 
          "000X0",
          "00X00",
          "0X000",
          "X0000",
          "pre-trained",
          "*****"]

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

# group by the type of layers inserted
group_0 = []
group_1 = []
group_2 = []
group_3 = []
group_4 = []
group_5 = []
group_pre = []
group_mixed = []

for x,y in enumerate(sorted_short_name):
        if(y[0]=='0' and y[1]=='0' and y[2]=='0' and y[3]=='0' and y[4]=='0'):
            group_0.append(x)
        elif(y[0]=='0' and y[1]=='0' and y[2]=='0' and y[3]=='0' and y[4]!='0'):
            group_1.append(x)
        elif(y[0]=='0' and y[1]=='0' and y[2]=='0' and y[3]!='0' and y[4]=='0'):
            group_2.append(x)
        elif(y[0]=='0' and y[1]=='0' and y[2]!='0' and y[3]=='0' and y[4]=='0'):
            group_3.append(x)
        elif(y[0]=='0' and y[1]!='0' and y[2]=='0' and y[3]=='0' and y[4]=='0'):
            group_4.append(x)
        elif(y[0]!='0' and y[1]=='0' and y[2]=='0' and y[3]=='0' and y[4]=='0'):
            group_5.append(x)
        elif(y == 'pre-trained'):
            group_pre.append(x)
        else:
            group_mixed.append(x)

## mAP50
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
group_mixed_mAP50 = []
for x in group_mixed:
    group_mixed_mAP50.append(sorted_mAP[x])

## instructions_retired
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
group_mixed_instructions = []
for x in group_mixed:
    group_mixed_instructions.append(instructions_retired[x])

# plot l2 misses vs. retired instructions
plt.figure(figsize=(18,6), dpi=120)
plt.scatter(group_0_instructions,group_0_mAP50,color=colors[0],label=labels[0])
plt.plot(group_1_instructions,group_1_mAP50,color=colors[1],label=labels[1],marker='o')
plt.plot(group_2_instructions,group_2_mAP50,color=colors[2],label=labels[2],marker='o')
plt.plot(group_3_instructions,group_3_mAP50,color=colors[3],label=labels[3],marker='o')
plt.plot(group_4_instructions,group_4_mAP50,color=colors[4],label=labels[4],marker='o')
plt.plot(group_5_instructions,group_5_mAP50,color=colors[5],label=labels[5],marker='o')
plt.scatter(group_pre_instructions,group_pre_mAP50,color=colors[6],label=labels[6])
plt.scatter(group_mixed_instructions,group_mixed_mAP50,color=colors[7],label=labels[7])

# plt.plot([group_0_instructions[0],group_1_instructions[0]],[group_0_mAP50[0],group_1_mAP50[0]],color=colors[1])
# plt.plot([group_0_instructions[0],group_2_instructions[0]],[group_0_mAP50[0],group_2_mAP50[0]],color=colors[2])
# plt.plot([group_0_instructions[0],group_3_instructions[0]],[group_0_mAP50[0],group_3_mAP50[0]],color=colors[3])
# plt.plot([group_0_instructions[0],group_4_instructions[0]],[group_0_mAP50[0],group_4_mAP50[0]],color=colors[4])
# plt.plot([group_0_instructions[0],group_5_instructions[0]],[group_0_mAP50[0],group_5_mAP50[0]],color=colors[5])

# plt.annotate(text='00000',fontsize=10,fontweight=600,color=colors[0],xy=[group_0_instructions[0],group_0_mAP50[0]+0.01e7],xytext=[group_0_instructions[0]-0.07e9,group_0_mAP50[0]+0.08e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[0]))
# plt.annotate(text='00001',fontsize=10,fontweight=600,color=colors[1],xy=[group_1_instructions[0],group_1_mAP50[0]+0.01e7],xytext=[group_1_instructions[0]-0.07e9,group_1_mAP50[0]+0.08e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[1]))
# plt.annotate(text='00002',fontsize=10,fontweight=600,color=colors[1],xy=[group_1_instructions[1],group_1_mAP50[1]+0.01e7],xytext=[group_1_instructions[1]-0.07e9,group_1_mAP50[1]+0.08e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[1]))
# plt.annotate(text='00003',fontsize=10,fontweight=600,color=colors[1],xy=[group_1_instructions[2],group_1_mAP50[2]+0.01e7],xytext=[group_1_instructions[2]-0.07e9,group_1_mAP50[2]+0.08e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[1]))
# plt.annotate(text='00010',fontsize=10,fontweight=600,color=colors[2],xy=[group_2_instructions[0],group_2_mAP50[0]-0.01e7],xytext=[group_2_instructions[0]-0.07e9,group_2_mAP50[0]-0.1e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[2]))
# plt.annotate(text='00020',fontsize=10,fontweight=600,color=colors[2],xy=[group_2_instructions[1],group_2_mAP50[1]-0.01e7],xytext=[group_2_instructions[1]-0.07e9,group_2_mAP50[1]-0.1e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[2]))
# plt.annotate(text='00030',fontsize=10,fontweight=600,color=colors[2],xy=[group_2_instructions[2],group_2_mAP50[2]-0.01e7],xytext=[group_2_instructions[2]-0.07e9,group_2_mAP50[2]-0.1e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[2]))
# plt.annotate(text='00100',fontsize=10,fontweight=600,color=colors[3],xy=[group_3_instructions[0],group_3_mAP50[0]-0.01e7],xytext=[group_3_instructions[0]+0.03e9,group_3_mAP50[0]-0.1e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[3]))
# plt.annotate(text='00200',fontsize=10,fontweight=600,color=colors[3],xy=[group_3_instructions[1],group_3_mAP50[1]-0.01e7],xytext=[group_3_instructions[1]+0.03e9,group_3_mAP50[1]-0.1e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[3]))
# plt.annotate(text='00300',fontsize=10,fontweight=600,color=colors[3],xy=[group_3_instructions[2],group_3_mAP50[2]-0.01e7],xytext=[group_3_instructions[2]+0.03e9,group_3_mAP50[2]-0.1e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[3]))
# plt.annotate(text='01000',fontsize=10,fontweight=600,color=colors[4],xy=[group_4_instructions[0],group_4_mAP50[0]+0.01e7],xytext=[group_4_instructions[0]+0.03e9,group_4_mAP50[0]+0.07e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[4]))
# plt.annotate(text='02000',fontsize=10,fontweight=600,color=colors[4],xy=[group_4_instructions[1],group_4_mAP50[1]+0.01e7],xytext=[group_4_instructions[1]+0.03e9,group_4_mAP50[1]+0.07e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[4]))
# plt.annotate(text='03000',fontsize=10,fontweight=600,color=colors[4],xy=[group_4_instructions[2],group_4_mAP50[2]+0.01e7],xytext=[group_4_instructions[2]+0.03e9,group_4_mAP50[2]+0.07e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[4]))
# plt.annotate(text='10000',fontsize=10,fontweight=600,color=colors[5],xy=[group_5_instructions[0],group_5_mAP50[0]+0.01e7],xytext=[group_5_instructions[0]+0.03e9,group_5_mAP50[0]+0.1e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[5]))
# plt.annotate(text='20000',fontsize=10,fontweight=600,color=colors[5],xy=[group_5_instructions[1],group_5_mAP50[1]+0.01e7],xytext=[group_5_instructions[1]-0.07e9,group_5_mAP50[1]+0.07e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[5]))
# plt.annotate(text='30000',fontsize=10,fontweight=600,color=colors[5],xy=[group_5_instructions[2],group_5_mAP50[2]-0.01e7],xytext=[group_5_instructions[2]-0.07e9,group_5_mAP50[2]-0.15e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[5]))
# plt.annotate(text='pre-trained',fontsize=10,fontweight=600,color=colors[6],xy=[group_pre_instructions[0],group_pre_mAP50[0]-0.01e7],xytext=[group_pre_instructions[0]+0.03e9,group_pre_mAP50[0]-0.1e7],
#     arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=5',color=colors[6]))

plt.grid(alpha=0.5)
# plt.xticks(np.arange(1.6e9, 2.8e9, 0.05e9))
# plt.yticks(np.arange(2.0e7, 3.5e7, 0.1e7))
plt.xlabel('retired instructions')
plt.ylabel('mAP50 (%)')
plt.legend()
plt.savefig('YOLO_mAP50-instructions_retired.pdf', bbox_inches='tight')
