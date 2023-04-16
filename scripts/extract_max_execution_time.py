from os import listdir
import pandas as pd
import numpy as np

data_folder = "../benchmark_results/image_classification/"
models_list = listdir(data_folder)
models_list.sort()
models = [folder+"/" for folder in models_list]

job_elapsed_seconds = []

for model in models:
    # read file
    data = pd.read_csv(data_folder+model+"timing.csv")
    # acquire metrics
    job_elapsed_seconds += [np.max(data["job_elapsed(seconds)"].values)]
    
print(np.max(job_elapsed_seconds))

