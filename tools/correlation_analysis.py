import numpy as np
import pandas as pd
import os
from pycls.datasets.imbalanced_cifar import IMBALANCECIFAR10
import matplotlib.pyplot as plt


score_files = [
# '/media/ntu/volume2/home/s121md302_06/workspace/code/temperature-as-uncertainty-public/experiments/tau_simclr_base/viz/tau_uncertainty.npy',
# '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/results/cifar-10/CIFAR10_SimCLR_losses_10.npy',
# '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/results/cifar-10/CIFAR10_SCAN_0.3_overall.npy',
# '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/results/cifar-10/CIFAR10_SimCLR_losses.npy',
# '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/results/cifar-10/CIFAR10_SimCLR_losses_10_temp_0.3.npy',
# '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/results/cifar-10/CIFAR10_SimCLR_losses_10_temp_0.3_overall.npy',
# '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/results/cifar-10/CIFAR10_SimCLR_losses_10_temp_0.3_overall_plus.npy',
# '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/results/cifar-10/CIFAR10_SimCLR_losses_10_temp_0.3_plus.npy',
# '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/results/cifar-10/CIFAR10_SimCLR_losses_10_temp_1.npy',
# '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/results/cifar-10/CIFAR10_SimCLR_losses_closest.npy',
# # '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/results/cifar-10/CIFAR10_VAE_losses.npy',
# '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/results/cifar-10/CIFAR10_diff_norm.npy',
'/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/results/cifar-10/CIFAR10_reverse_diff_norm.npy',

]


for score_file in score_files:
    score = np.load(score_file, allow_pickle=True)

    # score_df = df = pd.DataFrame(columns=[0.2, 0.5, 1, 2])
    score_df = df = pd.DataFrame(columns=[0.05, 0.1, 0.2, 0.5, 1, 2])
    # score_df = df = pd.DataFrame(columns=[0.2, 0.5, 1, 2])
    # score_df = df = pd.DataFrame(columns=[3.5, 7])
    # df = pd.DataFrame(columns=[0.2, 2])

    for col in range(1, 101):
    # for col in range(1, 31):
    # for col in range(51,
    # 86):
        row, score_row = [], []
        # for ratio in [0.2, 2]:
        # for ratio in [0.2, 0.5, 1, 2]:
        for ratio in [0.05, 0.1, 0.2, 0.5, 1, 2]:
        # for ratio in [3.5, 7]:
            file = 'plot_episode_yvalues.txt'
            trial = 'al_cifar10_random_'+str(ratio)+'percent_start_only__deterministic_trial'+str(col)
            # trial = 'al_cifar10_random_'+str(ratio)+'percent_im_start_only__deterministic_trial'+str(col)
            dir = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/CIFAR10/resnet18/'
            # dir = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/IMBALANCED_CIFAR10/resnet18/'
            path = os.path.join(dir, trial, file)
            initial_pool_file = 'lSet.npy'
            lSet = np.load(os.path.join(dir, trial, initial_pool_file), allow_pickle=True).astype(int)
            sum_score = score[lSet].sum()
            score_row.append(sum_score)
            with open(path) as f:
                lines = f.readlines()
                result = float(lines[0])
                row.append(result)
        # df = df.append([{0.2: row[0], 0.5: row[1], 1: row[2], 2: row[3]}], ignore_index=True)
        # score_df = score_df.append([{0.2: score_row[0], 0.5: score_row[1], 1: score_row[2], 2: score_row[3]}], ignore_index=True)
        df = df.append([{0.05: row[0], 0.1: row[1], 0.2: row[2], 0.5: row[3], 1: row[4], 2: row[5]}], ignore_index=True)
        score_df = score_df.append([{0.05: score_row[0], 0.1: score_row[1], 0.2: score_row[2], 0.5: score_row[3], 1: score_row[4], 2: score_row[5]}], ignore_index=True)
        # df = df.append([{3.5: row[0], 7: row[1]}], ignore_index=True)
        # df = df.append([{0.2: row[0], 2: row[1]}], ignore_index=True)

    # print(df)
    pd.options.display.float_format = "{:,.2f}".format
    # print(df.describe())
    idxmax = [df[column].idxmax() for column in df.columns]
    idxmean = [np.argsort(df[column])[int(len(df[column])//2)] for column in df.columns]
    idxmin = [df[column].idxmin() for column in df.columns]
    # print('idxmin', idxmin)
    # print('idxmean', idxmean)
    # print('idxmax', idxmax)
    # # workspace/code/init-pools-dal/output/CIFAR10/resnet18/al_cifar10_random_1percent_start_only_trial21
    print(score_df)
    print(score_file.split('/')[-1])
    print(df.corrwith(score_df, axis = 0))

    for column in df.columns:
        plt.scatter(score_df[column], df[column])
        plt.title(column)
        plt.show()


