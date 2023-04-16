from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

extra_data = pd.read_csv("sort.csv")
inference_time = extra_data["inference_time"].values
l2_misses = extra_data["l2_misses"].values
bandwidth = extra_data["bandwidth"].values
mAP50 = extra_data["mAP50"].values

extra_data = pd.read_csv("extra_benchmarks_full.csv")
models = extra_data["short_name"].values
models_rank = []

for model in models:
    models_rank.append((np.argwhere(inference_time == model)[0] + 
                        np.argwhere(l2_misses == model)[0] + 
                        np.argwhere(bandwidth == model)[0]+ 
                        np.argwhere(mAP50 == model)[0])[0])
    # print(model, (np.argwhere(inference_time == model)[0] + 
    #               np.argwhere(l2_misses == model)[0] + 
    #               np.argwhere(bandwidth == model)[0]+ 
    #               np.argwhere(mAP50 == model)[0])[0])

association = zip(models, models_rank)
association_sorted = sorted(association, key=lambda x:x[1])
sorted_results = zip(*association_sorted)
sorted_models, sorted_models_rank = [list(x) for x in sorted_results]

with open(r'sort.csv', mode='a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["total_rank"])
    for i in sorted_models:
        writer.writerow(i)