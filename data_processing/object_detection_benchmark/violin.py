from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

colors = ["purple", "c", "chocolate", "orange", "green", "royalblue", "red"]
labels = ["00000",
          "0000X", 
          "000X*",
          "00X**",
          "0X***",
          "X****",
          "pre-trained"]


def add_label(add_labels, violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    add_labels.append((mpatches.Patch(color=color), label))

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
index = 0
data_folder = "benchmarks/"

job_elapsed = np.zeros((4942,57))
instructions_retired = np.zeros((4942,57))
l1_misses = np.zeros((4942,57))
l1_miss_ratio = np.zeros((4942,57))
l2_misses = np.zeros((4942,57))
l2_miss_ratio = np.zeros((4942,57))
cpu_clock_count = np.zeros((4942,57))

bandwidth = np.zeros((4942,57))
instructions_per_second = np.zeros((4942,57))
instructions_per_clock = np.zeros((4942,57))

for model in sorted_models:
    data = pd.read_csv(data_folder+model+"/"+"timing.csv")
    job_elapsed[:,index] = np.array(data["job_elapsed(seconds)"].values)
    instructions_retired[:,index] = np.array(data["instructions_retired"].values)
    l1_misses[:,index] = np.array(data["job_l1_misses"].values)
    l1_miss_ratio[:,index] = np.array(data["job_l1_miss_ratio(%%)"].values)
    l2_misses[:,index] = np.array(data["job_l2_misses"].values)
    l2_miss_ratio[:,index] = np.array(data["job_l2_miss_ratio(%%)"].values)
    cpu_clock_count[:,index] = np.array(data["cpu_clock_count"].values)

    bandwidth[:,index] = np.divide(l2_misses[:,index], job_elapsed[:,index])
    instructions_per_second[:,index] = np.divide(instructions_retired[:,index], job_elapsed[:,index])
    instructions_per_clock[:,index] = np.divide(instructions_retired[:,index], cpu_clock_count[:,index])

    index = index + 1

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

# plot violin of inference_time
group_0_time = []
for x in group_0:
    group_0_time.append(job_elapsed[:,x])
group_1_time = []
for x in group_1:
    group_1_time.append(job_elapsed[:,x])
group_2_time = []
for x in group_2:
    group_2_time.append(job_elapsed[:,x])
group_3_time = []
for x in group_3:
    group_3_time.append(job_elapsed[:,x])
group_4_time = []
for x in group_4:
    group_4_time.append(job_elapsed[:,x])
group_5_time = []
for x in group_5:
    group_5_time.append(job_elapsed[:,x])
group_pre_time = []
for x in group_pre:
    group_pre_time.append(job_elapsed[:,x])

plt.figure(figsize=(16,20), dpi=240)

loop = 0
add_labels_time = []
group_loop = [group_0, group_1, group_2, group_3, group_4, group_5, group_pre]
metric_loop = [group_0_time, group_1_time, group_2_time, group_3_time, group_4_time, group_5_time, group_pre_time]
for a,b in zip(metric_loop, group_loop):
    violin = plt.violinplot(a, positions=b, widths=1.2, vert=True, showmedians=True, showextrema=True, quantiles=[[0.99]]*len(b))
    for patch in violin['bodies']:
        patch.set_facecolor(colors[loop])
        patch.set_edgecolor(colors[loop])
        patch.set_alpha(0.3)
    for partname in ('cbars','cmins','cmaxes','cmedians','cquantiles'):
        vp = violin[partname]
        vp.set_edgecolor(colors[loop])
        vp.set_linewidth(0.5)
    add_label(add_labels_time, violin, labels[loop])
    loop = loop + 1

plt.legend(*zip(*add_labels_time), loc=2)
plt.grid(alpha=0.5)
plt.xticks(range(0,57), sorted_short_name, rotation=75)
plt.yticks(np.arange(1.22, 2.46, 0.02))
plt.xlabel('models')
plt.ylabel('inference time (s)')
# plt.title('inference time vs. models')
plt.savefig('YOLO_inference_time-models_violin.pdf', bbox_inches='tight')

# plot violin of l2_misses
group_0_l2_misses = []
for x in group_0:
    group_0_l2_misses.append(l2_misses[:,x])
group_1_l2_misses = []
for x in group_1:
    group_1_l2_misses.append(l2_misses[:,x])
group_2_l2_misses = []
for x in group_2:
    group_2_l2_misses.append(l2_misses[:,x])
group_3_l2_misses = []
for x in group_3:
    group_3_l2_misses.append(l2_misses[:,x])
group_4_l2_misses = []
for x in group_4:
    group_4_l2_misses.append(l2_misses[:,x])
group_5_l2_misses = []
for x in group_5:
    group_5_l2_misses.append(l2_misses[:,x])
group_pre_l2_misses = []
for x in group_pre:
    group_pre_l2_misses.append(l2_misses[:,x])

plt.figure(figsize=(16,20), dpi=240)

loop = 0
add_labels_l2_misses = []
group_loop = [group_0, group_1, group_2, group_3, group_4, group_5, group_pre]
metric_loop = [group_0_l2_misses, group_1_l2_misses, group_2_l2_misses, group_3_l2_misses, group_4_l2_misses, group_5_l2_misses, group_pre_l2_misses]
for a,b in zip(metric_loop, group_loop):
    violin = plt.violinplot(a, positions=b, widths=1.2, vert=True, showmedians=True, showextrema=True, quantiles=[[0.99]]*len(b))
    for patch in violin['bodies']:
        patch.set_facecolor(colors[loop])
        patch.set_edgecolor(colors[loop])
        patch.set_alpha(0.3)
    for partname in ('cbars','cmins','cmaxes','cmedians','cquantiles'):
        vp = violin[partname]
        vp.set_edgecolor(colors[loop])
        vp.set_linewidth(0.5)
    add_label(add_labels_l2_misses, violin, labels[loop])
    loop = loop + 1

plt.legend(*zip(*add_labels_l2_misses), loc=2)
plt.grid(alpha=0.5)
plt.xticks(range(0,57), sorted_short_name, rotation=75)
plt.yticks(np.arange(2.12e7, 3.34e7, 0.02e7))
plt.xlabel('models')
plt.ylabel('l2 misses')
# plt.title('l2 misses vs. models')
plt.savefig('YOLO_l2_misses-models_violin.pdf', bbox_inches='tight')

# plot violin of instructions_per_clock
group_0_instructions_per_clock = []
for x in group_0:
    group_0_instructions_per_clock.append(instructions_per_clock[:,x])
group_1_instructions_per_clock = []
for x in group_1:
    group_1_instructions_per_clock.append(instructions_per_clock[:,x])
group_2_instructions_per_clock = []
for x in group_2:
    group_2_instructions_per_clock.append(instructions_per_clock[:,x])
group_3_instructions_per_clock = []
for x in group_3:
    group_3_instructions_per_clock.append(instructions_per_clock[:,x])
group_4_instructions_per_clock = []
for x in group_4:
    group_4_instructions_per_clock.append(instructions_per_clock[:,x])
group_5_instructions_per_clock = []
for x in group_5:
    group_5_instructions_per_clock.append(instructions_per_clock[:,x])
group_pre_instructions_per_clock = []
for x in group_pre:
    group_pre_instructions_per_clock.append(instructions_per_clock[:,x])

plt.figure(figsize=(16,20), dpi=240)

loop = 0
add_labels_instructions_per_clock = []
group_loop = [group_0, group_1, group_2, group_3, group_4, group_5, group_pre]
metric_loop = [group_0_instructions_per_clock, group_1_instructions_per_clock, group_2_instructions_per_clock, group_3_instructions_per_clock, group_4_instructions_per_clock, group_5_instructions_per_clock, group_pre_instructions_per_clock]
for a,b in zip(metric_loop, group_loop):
    violin = plt.violinplot(a, positions=b, widths=1.2, vert=True, showmedians=True, showextrema=True, quantiles=[[0.99]]*len(b))
    for patch in violin['bodies']:
        patch.set_facecolor(colors[loop])
        patch.set_edgecolor(colors[loop])
        patch.set_alpha(0.3)
    for partname in ('cbars','cmins','cmaxes','cmedians','cquantiles'):
        vp = violin[partname]
        vp.set_edgecolor(colors[loop])
        vp.set_linewidth(0.5)
    add_label(add_labels_instructions_per_clock, violin, labels[loop])
    loop = loop + 1

plt.legend(*zip(*add_labels_instructions_per_clock), loc=3)
plt.grid(alpha=0.5)
plt.xticks(range(0,57), sorted_short_name, rotation=75)
plt.yticks(np.arange(0.80, 0.94, 0.002))
plt.xlabel('models')
plt.ylabel('instructions per clock (IPC)')
# plt.title('instructions per clock (IPC) vs. models')
plt.savefig('YOLO_instructions_per_clock-models_violin.pdf', bbox_inches='tight')

# first data vs. maximum data of inference_time
job_elapsed_median = np.median(job_elapsed, axis=0)
job_elapsed_first = job_elapsed[0,:]
job_elapsed_eliminate_first = np.delete(job_elapsed, 0, axis=0)
job_elapsed_eliminate_first_max = np.max(job_elapsed_eliminate_first, axis=0)

first_divide_median = np.divide(job_elapsed_first, job_elapsed_median)
eliminate_first_max_divide_median = np.divide(job_elapsed_eliminate_first_max, job_elapsed_median)

plt.figure(figsize=(15,6), dpi=80)
plt.bar(range(0,57), first_divide_median, width=0.5, fc='orange', label='the data of the first execution / median')
plt.bar(range(0,57), eliminate_first_max_divide_median, width=0.3, fc='cornflowerblue', label='the maximum data after eliminate the first execution / median')
plt.legend()
plt.xticks(range(0,57), sorted_short_name, rotation=75, fontsize=8)
plt.ylim(1.0,1.21)
plt.xlabel('models')
plt.ylabel('ratio of inference time')
# plt.title('ratio of inference time vs. models')
plt.savefig('YOLO_inference_time-models_ratio.pdf', bbox_inches='tight')

# first data vs. maximum data of l2_misses
l2_misses_median = np.median(l2_misses, axis=0)
l2_misses_first = l2_misses[0,:]
l2_misses_eliminate_first = np.delete(l2_misses, 0, axis=0)
l2_misses_eliminate_first_max = np.max(l2_misses_eliminate_first, axis=0)

first_divide_median = np.divide(l2_misses_first, l2_misses_median)
eliminate_first_max_divide_median = np.divide(l2_misses_eliminate_first_max, l2_misses_median)

plt.figure(figsize=(15,6), dpi=80)
plt.bar(range(0,57), eliminate_first_max_divide_median, width=0.5, fc='cornflowerblue', label='the maximum data after eliminate the first execution / median')
plt.bar(range(0,57), first_divide_median, width=0.3, fc='orange', label='the data of the first execution / median')
plt.legend()
plt.xticks(range(0,57), sorted_short_name, rotation=75, fontsize=8)
plt.ylim(1.0,1.075)
plt.xlabel('models')
plt.ylabel('ratio of l2 misses')
# plt.title('ratio of l2 misses vs. models')
plt.savefig('YOLO_l2_misses-models_ratio.pdf', bbox_inches='tight')

# first data vs. maximum data of l1 misses
l1_misses_median = np.median(l1_misses, axis=0)
l1_misses_first = l1_misses[0,:]
l1_misses_eliminate_first = np.delete(l1_misses, 0, axis=0)
l1_misses_eliminate_first_max = np.max(l1_misses_eliminate_first, axis=0)

first_divide_median = np.divide(l1_misses_first, l1_misses_median)
eliminate_first_max_divide_median = np.divide(l1_misses_eliminate_first_max, l1_misses_median)

plt.figure(figsize=(15,6), dpi=80)
plt.bar(range(0,57), first_divide_median, width=0.5, fc='orange', label='the data of the first execution / median')
plt.bar(range(0,57), eliminate_first_max_divide_median, width=0.3, fc='cornflowerblue', label='the maximum data after eliminate the first execution / median')
plt.legend()
plt.xticks(range(0,57), sorted_short_name, rotation=75, fontsize=8)
plt.ylim(1.0,1.25)
plt.xlabel('models')
plt.ylabel('ratio of l1-d misses')
# plt.title('ratio of l1 misses vs. models')
plt.savefig('YOLO_l1_misses-models_ratio.pdf', bbox_inches='tight')
