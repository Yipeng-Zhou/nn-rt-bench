import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

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

# extract the data from the folder "benchmarks" and plot
index = 0
data_folder = "benchmarks/"
plt.figure(figsize=(12,6), dpi=120)
for model in models:
    # read file
    data = pd.read_csv(data_folder+model+"/"+"perf.csv")
    # extract column
    l2_references = np.divide(data["l2_references"], data["samples"])
    l2_references = np.divide(l2_references, l2_references.iloc[-1]).values
    l2_refills = np.divide(data["l2_refills"], data["samples"])
    l2_refills = np.divide(l2_refills, l2_refills.iloc[-1]).values
    l1_references = np.divide(data["l1_references"], data["samples"])
    l1_references = np.divide(l1_references, l1_references.iloc[-1]).values
    l1_refills = np.divide(data["l1_refills"], data["samples"])
    l1_refills = np.divide(l1_refills, l1_refills.iloc[-1]).values
    inst_retired = np.divide(data["inst_retired"], data["samples"])
    inst_retired = np.divide(inst_retired, inst_retired.iloc[-1]).values
    
    # plot
    x = np.arange(len(inst_retired))
    # plt.plot(x, l1_references, label="L1 References")
    # plt.plot(x, l1_refills, label="L1 Refills")
    # plt.plot(x, l2_references, label="LLC References")
    plt.plot(x, l2_refills, label=labels[index], color=colors[index])
    # plt.plot(x, inst_retired, label="Inst. Retired")
    index = index + 1

plt.xlabel("Time (10ms)")
plt.ylabel("Normalized l2_refills cumulative activation")
plt.legend()
plt.title('Normalized l2_refills cumulative activation')
plt.savefig("NCA_l2_refills.png", bbox_inches='tight')