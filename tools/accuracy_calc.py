import numpy as np
import pandas as pd
import os

# df = pd.DataFrame(columns=[0.2, 0.5, 1, 2])
df = pd.DataFrame(columns=[0.05, 0.1, 0.2, 0.5])
# df = pd.DataFrame(columns=[3.5, 7])
# df = pd.DataFrame(columns=[0.2, 2])

for col in range(1, 101):
# for col in range(1, 31):
# for col in range(51,
# 86):
    row = []
    # for ratio in [0.2, 2]:
    # for ratio in [0.2, 0.5, 1, 2]:
    for ratio in [0.05, 0.1, 0.2, 0.5]:
    # for ratio in [3.5, 7]:
        file = 'plot_episode_yvalues.txt'
        trial = 'al_cifar10_random_'+str(ratio)+'percent_start_only__deterministic_trial'+str(col)
        # trial = 'al_cifar10_random_'+str(ratio)+'percent_im_start_only__deterministic_trial'+str(col)
        dir = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/CIFAR10/resnet18/'
        # dir = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/IMBALANCED_CIFAR10/resnet18/'
        path = os.path.join(dir, trial, file)
        with open(path) as f:
            lines = f.readlines()
            result = float(lines[0])
            row.append(result)
    # df = df.append([{0.2: row[0], 0.5: row[1], 1: row[2], 2: row[3]}], ignore_index=True)
    df = df.append([{0.05: row[0], 0.1: row[1], 0.2: row[2], 0.5: row[3]}], ignore_index=True)
    # df = df.append([{3.5: row[0], 7: row[1]}], ignore_index=True)
    # df = df.append([{0.2: row[0], 2: row[1]}], ignore_index=True)

print(df)
pd.options.display.float_format = "{:,.2f}".format
print(df.describe())
idxmax = [df[column].idxmax() for column in df.columns]
idxmean = [np.argsort(df[column])[int(len(df[column])//2)] for column in df.columns]
idxmin = [df[column].idxmin() for column in df.columns]
print('idxmin', idxmin)
print('idxmean', idxmean)
print('idxmax', idxmax)
# workspace/code/init-pools-dal/output/CIFAR10/resnet18/al_cifar10_random_1percent_start_only_trial21