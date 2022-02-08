import argparse
import glob
import itertools
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import scipy.stats
import seaborn as sns
import torch
import torch.nn as nn
from pandarallel import pandarallel
from scipy import spatial
from torchvision import transforms
from tqdm import tqdm

import pycls.core.builders as model_builder
import pycls.utils.checkpoint as cu
from pycls.core.config import cfg
from pycls.datasets.data import Data

pandarallel.initialize()

def argparser():
    parser = argparse.ArgumentParser(description='Active Learning Cold Start Benchmark')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    return parser


def plot_class_dist(labels, index_to_class_dict, acquisition='random', ratio=0.01, title=None, ylim=(None, None)):
    classes, counts = np.unique(labels, return_counts=True)
    df_rel = pd.DataFrame(columns=['classes', 'counts'])
    df_rel['classes'], df_rel['counts'] = classes, counts
    figsize = [5, 5]
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.autolayout"] = True
    df_rel.plot(x='classes', y='counts', kind='bar', stacked=True,
                title=title,
                legend=None,
                figsize=figsize, colormap='Reds_r',
                xlabel=None,
                xticks=None,
                width=0.9
                )
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.ylim(ylim)
    plt.savefig(os.path.join(plot_save_dir, dataset, acquisition + '_' + str(ratio) + '_distribution_histogram.png'))
    plt.show()


def read_dir_by_acquisition(path):
    file_list = glob.glob(path + '*.out')
    file_info = {}
    for file in file_list:
        file_info[file] = {'trial': file.split('-')[1],
                           'ratio': file.split('-p')[-1].split('-')[0],
                           'acquisition': file.split('-')[-1].split('.out')[0]
                           }

        # debug only
        if file.split('-')[-1].split('.out')[0] == 'vaal':
            print('vaal', file.split('-p')[-1].split('-')[0])

        f = open(file, "r")
        lines = f.readlines()

        al_sorted_idx_file = os.path.join(df_save_dir, train_data.__class__.__name__,
                                          file.split('-')[-1].split('.out')[0] + '_sorted_idx.npy')
        sorted_idx = np.load(al_sorted_idx_file)
        al_sorted_idx = sorted_idx[:int(len(sorted_idx) * float(file_info[file]['ratio']))]

        for idx, line in enumerate(lines):
            if line.endswith('test samples\n'):
                file_info[file]['indices'] = al_sorted_idx.tolist()
            if line.startswith('AVERAGE | AUC'):
                file_info[file]['auc'] = line.split('= ')[-1].split('\n')[0]

    return file_info


def read_dir(path):
    file_list = glob.glob(path + '*.out')
    file_info = {}
    for file in file_list:
        file_info[file] = {'trial': file.split('-')[1],
                           'ratio': file.split('-p')[-1].split('.out')[0],
                           'acquisition': 'random'
                           }

        f = open(file, "r")
        lines = f.readlines()

        for idx, line in enumerate(lines):
            if line.endswith('test samples\n'):
                file_info[file]['indices'] = [int(index) for index in list(lines[idx + 1][1:-2].split(','))]
            if line.startswith('AVERAGE | AUC'):
                file_info[file]['auc'] = line.split('= ')[-1].split('\n')[0]

    return file_info


def load_model(cfg, checkpoint_file):
    model = model_builder.build_model(cfg)
    model = cu.load_checkpoint(checkpoint_file, model)
    model.eval()
    return model


def load_train_data(cfg, data_obj):
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('..'), cfg.DATASET.ROOT_DIR)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    return train_data, train_size


def load_test_data(cfg, data_obj):
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('..'), cfg.DATASET.ROOT_DIR)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)
    return test_data, test_size


def calculate_global_score(al, cfg, dataObj, model, dataset):
    if al == 'uncertainty':
        score = uncertainty(cfg, dataObj=dataObj, model=model, dataset=dataset)
    elif al == 'coreset':
        raise NotImplementedError('Use coreset score calculation.')
    elif al == 'margin':
        score = margin(cfg, dataObj=dataObj, model=model, dataset=dataset)
    elif al == 'bald':
        score = bald(cfg, dataObj=dataObj, model=model, dataset=dataset)
    elif al == 'consistency':
        score = consistency(cfg, dataObj=dataObj, model=model, dataset=dataset)
    elif al == 'vaal':
        score = vaal(cfg, dataObj=dataObj, model=model, dataset=dataset)
    return score


def uncertainty(cfg, dataObj, model, dataset):
    """
    Implements the uncertainty principle as a acquisition function.
    """
    if gpu is True:
        num_classes = cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(
            model.training)

        clf = model.cuda()

        u_ranks = []
        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     print("Loading the model in data parallel where num_GPUS: {}".format(self.cfg.NUM_GPUS))
        #     clf = torch.nn.DataParallel(clf, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
        #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        uSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(dataset)),
                                                     batch_size=int(cfg.TRAIN.BATCH_SIZE), data=dataset)

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)

                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank = temp_u_rank * torch.log2(temp_u_rank)
                temp_u_rank = -1 * torch.sum(temp_u_rank, dim=1)
                u_ranks.append(temp_u_rank.detach().cpu().numpy())
        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]
        Path(os.path.join(df_save_dir, dataset.__class__.__name__)).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'uncertainty_u_ranks.npy'), u_ranks)
    else:
        u_ranks = np.load(os.path.join(df_save_dir, dataset.__class__.__name__, 'uncertainty_u_ranks.npy'))

    # index of u_ranks serve as key to refer in u_idx
    print(f"u_ranks.shape: {u_ranks.shape}")
    # we add -1 for reversing the sorted array
    sorted_idx = np.argsort(u_ranks)[
                 ::-1]  # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
    np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'uncertainty_sorted_idx.npy'), sorted_idx)
    return u_ranks


def consistency(cfg, dataObj, model, dataset):
    if gpu is True:
        # config dataloaders
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(
            model.training)

        clf = model.cuda()

        duplicates = 50
        duplicate_pred = []
        for i in range(duplicates):
            u_ranks = []
            # if self.cfg.TRAIN.DATASET == "IMAGENET":
            #     print("Loading the model in data parallel where num_GPUS: {}".format(self.cfg.NUM_GPUS))
            #     clf = torch.nn.DataParallel(clf, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
            #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
            # else:
            uSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(dataset)),
                                                         batch_size=int(cfg.TRAIN.BATCH_SIZE),
                                                         data=dataset)

            n_uLoader = len(uSetLoader)
            print("len(uSetLoader): {}".format(n_uLoader))
            for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
                with torch.no_grad():
                    x_u = x_u.cuda(0)

                    temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                    u_ranks.append(temp_u_rank.detach().cpu().numpy())
            u_ranks = np.concatenate(u_ranks, axis=0)
            duplicate_pred.append(u_ranks)
        duplicate_pred = np.asarray(duplicate_pred)
        Path(os.path.join(df_save_dir, dataset.__class__.__name__)).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'consistency.npy'), duplicate_pred)
    else:
        duplicate_pred = np.load(os.path.join(df_save_dir, dataset.__class__.__name__, 'consistency.npy'))

    var = np.var(duplicate_pred, axis=0).sum(axis=1)
    sorted_idx = np.argsort(var)[
                 ::-1]  # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
    np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'consistency_sorted_idx.npy'), sorted_idx)
    return var


def margin(cfg, dataObj, model, dataset):
    """
    Implements the uncertainty principle as a acquisition function.
    """
    if gpu is True:
        num_classes = cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(
            model.training)

        clf = model.cuda()

        u_ranks = []
        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     print("Loading the model in data parallel where num_GPUS: {}".format(self.cfg.NUM_GPUS))
        #     clf = torch.nn.DataParallel(clf, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
        #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        uSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(dataset)),
                                                     batch_size=int(cfg.TRAIN.BATCH_SIZE), data=dataset)

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)

                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank, _ = torch.sort(temp_u_rank, descending=True)
                difference = temp_u_rank[:, 0] - temp_u_rank[:, 1]
                # for code consistency across uncertainty, entropy methods i.e., picking datapoints with max value
                difference = -1 * difference
                u_ranks.append(difference.detach().cpu().numpy())
        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]
        Path(os.path.join(df_save_dir, dataset.__class__.__name__)).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'margin_u_ranks.npy'), u_ranks)
    else:
        u_ranks = np.load(os.path.join(df_save_dir, dataset.__class__.__name__, 'margin_u_ranks.npy'))

    # index of u_ranks serve as key to refer in u_idx
    print(f"u_ranks.shape: {u_ranks.shape}")
    # we add -1 for reversing the sorted array
    sorted_idx = np.argsort(u_ranks)[
                 ::-1]  # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
    np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'margin_sorted_idx.npy'), sorted_idx)
    return u_ranks


def vaal(cfg, dataObj, model, dataset):
    if gpu is True:
        clf = model.cuda()
        lSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(train_data)),
                                                     batch_size=int(cfg.TRAIN.BATCH_SIZE),
                                                     data=train_data)
        uSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(dataset)),
                                                     batch_size=int(cfg.TRAIN.BATCH_SIZE),
                                                     data=dataset)

        out = []
        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)
                out.append(clf(x_u))
        u_out = torch.cat(out)

        out = []
        n_lLoader = len(lSetLoader)
        print("len(n_lLoader): {}".format(n_lLoader))
        for i, (x_u, _) in enumerate(tqdm(lSetLoader, desc="lSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)
                out.append(clf(x_u))
        l_out = torch.cat(out)

        score = []
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for u in u_out:
            u_score = (1 - cos(u, l_out)).sum()  # cosine distance
            score.append(u_score.detach().cpu().numpy())
        score = np.array(score)
        Path(os.path.join(df_save_dir, dataset.__class__.__name__)).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'vaal_score.npy'), score)
    else:
        score = np.load(os.path.join(df_save_dir, dataset.__class__.__name__, 'vaal_score.npy'))
    sorted_idx = np.argsort(score)[
                 ::-1]  # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
    np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'vaal_sorted_idx.npy'), sorted_idx)
    return score


def get_predictions(clf_model, dataObj, idx_set, dataset):
    clf_model.cuda()
    # Used by bald acquisition
    # if self.cfg.TRAIN.DATASET == "IMAGENET":
    #     tempIdxSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=idx_set, isDistributed=False, isShuffle=False, isVaalSampling=False)
    # else:
    tempIdxSetLoader = dataObj.getSequentialDataLoader(indexes=np.array(idx_set),
                                                       batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
                                                       data=dataset)

    preds = []
    for i, (x, _) in enumerate(tqdm(tempIdxSetLoader, desc="Collecting predictions in get_predictions function")):
        with torch.no_grad():
            x = x.cuda()
            x = x.type(torch.cuda.FloatTensor)

            temp_pred = clf_model(x)

            # To get probabilities
            temp_pred = torch.nn.functional.softmax(temp_pred, dim=1)
            preds.append(temp_pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    return preds


def bald(cfg, dataObj, model, dataset):
    if gpu is True:
        "Implements BALD acquisition function where we maximize information gain."
        clf_model = model
        clf_model.cuda()

        assert cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS != 0, "Expected dropout iterations > 0."

        # Set Batchnorm in eval mode whereas dropout in train mode
        clf_model.train()
        for m in clf_model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        uSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(dataset)),
                                                     batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
                                                     data=dataset)

        n_uPts = len(dataset)
        # Source Code was in tensorflow
        # To provide same readability we use same variable names where ever possible
        # Original TF-Code: https://github.com/Riashat/Deep-Bayesian-Active-Learning/blob/master/MC_Dropout_Keras/Dropout_Bald_Q10_N1000_Paper.py#L223

        # Heuristic: G_X - F_X
        score_All = np.zeros(shape=(n_uPts, cfg.MODEL.NUM_CLASSES))
        all_entropy_dropout = np.zeros(shape=(n_uPts))

        for d in tqdm(range(cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS), desc="Dropout Iterations"):
            dropout_score = get_predictions(clf_model=clf_model, dataObj=dataObj, idx_set=np.arange(len(dataset)),
                                            dataset=dataset)

            score_All += dropout_score

            # computing F_x
            dropout_score_log = np.log2(dropout_score + 1e-6)  # Add 1e-6 to avoid log(0)
            Entropy_Compute = -np.multiply(dropout_score, dropout_score_log)
            Entropy_per_Dropout = np.sum(Entropy_Compute, axis=1)

            all_entropy_dropout += Entropy_per_Dropout

        Avg_Pi = np.divide(score_All, cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS)
        Log_Avg_Pi = np.log2(Avg_Pi + 1e-6)
        Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
        Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
        G_X = Entropy_Average_Pi
        Average_Entropy = np.divide(all_entropy_dropout, cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS)
        F_X = Average_Entropy
        U_X = G_X - F_X
        Path(os.path.join(df_save_dir, dataset.__class__.__name__)).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'bald_U_X.npy'), U_X)
    else:
        U_X = np.load(os.path.join(df_save_dir, dataset.__class__.__name__, 'bald_U_X.npy'))

    print("U_X.shape: ", U_X.shape)
    sorted_idx = np.argsort(U_X)[
                 ::-1]  # argsort helps to return the indices of u_scores such that their corresponding values are sorted.
    np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'bald_sorted_idx.npy'), sorted_idx)
    return U_X


def coreset(cfg, dataObj, model, dataset):
    if gpu is True:
        clf = model.cuda()

        uSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(dataset)),
                                                     batch_size=int(cfg.TRAIN.BATCH_SIZE),
                                                     data=dataset)
        out = []
        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)
                out.append(clf(x_u))
        out = torch.cat(out)
        out = out.detach().cpu().numpy()
        Path(os.path.join(df_save_dir, dataset.__class__.__name__)).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'coreset_out.npy'), out)
    else:
        out = np.load(os.path.join(df_save_dir, dataset.__class__.__name__, 'coreset_out.npy'))
    return out


def calculate_coreset_score(indices):
    score = .0
    time = datetime.now()
    for idx, pair in enumerate(itertools.combinations(indices, 2)):
        output = spatial.distance.cosine(features[pair[0]], features[pair[1]])
        score += output
        if idx % 100000 == 0:
            print(idx, datetime.now() - time)

    return score


def calculate_local_score(indices):
    local_score = np.sum([score[index] for index in indices])
    return local_score


def get_coreset_index(cfg, dataObj, model, dataset, budget):
    clf = model.cuda()
    lSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(train_data)),
                                                 batch_size=int(cfg.TRAIN.BATCH_SIZE),
                                                 data=train_data)
    uSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(dataset)), batch_size=int(cfg.TRAIN.BATCH_SIZE),
                                                 data=dataset)

    out = []
    n_uLoader = len(uSetLoader)
    print("len(uSetLoader): {}".format(n_uLoader))
    for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
        with torch.no_grad():
            x_u = x_u.cuda(0)
            out.append(clf(x_u))
    u_out = torch.cat(out)

    out = []
    n_lLoader = len(lSetLoader)
    print("len(n_lLoader): {}".format(n_lLoader))
    for i, (x_u, _) in enumerate(tqdm(lSetLoader, desc="lSet Activations")):
        with torch.no_grad():
            x_u = x_u.cuda(0)
            out.append(clf(x_u))
    l_out = torch.cat(out)
    print("Solving K Center Greedy Approach")
    start = time.time()
    greedy_indexes, remainSet = greedy_k_center(labeled=l_out, unlabeled=u_out, budget=budget)
    np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'coreset_sorted_idx.npy'), greedy_indexes)
    return None


def greedy_k_center(labeled, unlabeled, budget):
    greedy_indices = [None for i in range(budget)]
    greedy_indices_counter = 0
    # move cpu to gpu
    # labeled = torch.from_numpy(labeled).cuda(0)
    # unlabeled = torch.from_numpy(unlabeled).cuda(0)

    print(f"[GPU] Labeled.shape: {labeled.shape}")
    print(f"[GPU] Unlabeled.shape: {unlabeled.shape}")
    # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
    st = time.time()
    min_dist, _ = torch.min(gpu_compute_dists(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), dim=0)
    min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))
    print(f"time taken: {time.time() - st} seconds")

    temp_range = 500
    dist = np.empty((temp_range, unlabeled.shape[0]))
    for j in tqdm(range(1, labeled.shape[0], temp_range), desc="Getting first farthest index"):
        if j + temp_range < labeled.shape[0]:
            dist = gpu_compute_dists(labeled[j:j + temp_range, :], unlabeled)
        else:
            dist = gpu_compute_dists(labeled[j:, :], unlabeled)

        min_dist = torch.cat((min_dist, torch.min(dist, dim=0)[0].reshape((1, min_dist.shape[1]))))

        min_dist = torch.min(min_dist, dim=0)[0]
        min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))

    # iteratively insert the farthest index and recalculate the minimum distances:
    _, farthest = torch.max(min_dist, dim=1)

    min_dist = min_dist.T[np.setdiff1d(range(min_dist.shape[1]), farthest.item())].T

    greedy_indices[greedy_indices_counter] = farthest.item()
    greedy_indices_counter += 1

    amount = budget - 1

    for i in tqdm(range(amount), desc="Constructing Active set"):
        remained_u_index = np.setdiff1d(range(len(unlabeled)), np.array([x for x in greedy_indices if x is not None]))
        dist = gpu_compute_dists(
            unlabeled[greedy_indices[greedy_indices_counter - 1], :].reshape((1, unlabeled.shape[1])),
            unlabeled[remained_u_index])

        min_dist = torch.cat((min_dist, dist))

        min_dist, _ = torch.min(min_dist, dim=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        _, farthest = torch.max(min_dist, dim=1)
        min_dist = min_dist.T[np.setdiff1d(range(min_dist.shape[1]), farthest.item())].T
        greedy_indices[greedy_indices_counter] = remained_u_index[farthest.item()]
        greedy_indices_counter += 1

    remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
    remainSet = np.array(list(remainSet))
    return greedy_indices, remainSet


def gpu_compute_dists(M1, M2):
    """
    Computes L2 norm square on gpu
    Assume
    M1: M x D matrix
    M2: N x D matrix

    output: M x N matrix
    """
    # print(f"Function call to gpu_compute dists; M1: {M1.shape} and M2: {M2.shape}")
    M1_norm = (M1 ** 2).sum(1).reshape(-1, 1)

    M2_t = torch.transpose(M2, 0, 1)
    M2_norm = (M2 ** 2).sum(1).reshape(1, -1)
    dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
    return dists


if __name__ == '__main__':
    cfg.merge_from_file(argparser().parse_args().cfg_file)
    dataset = cfg.DATASET.NAME

    # Define DATA_DIR: logs dir
    # checkpoint_file: model checkpoint
    # plot_save_dir: dir to save plots
    # df_save_dir: dir to save df
    plot_save_dir = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/plot/'
    df_save_dir = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/df/'
    if dataset == 'CIFAR10_REVERSE':
        DATA_DIR = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/cifar10_random_selection_wt_imagenet/logs/'
        DATA_DIR_ACTIVE = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/cifar10_active_selection_wt_imagenet/logs/'
        # checkpoint_file = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/CIFAR10_REVERSE/resnet18/reverse_trial1/episode_0/vlBest_acc_78_model_epoch_0180.pyth'
        checkpoint_file = os.path.join(df_save_dir, 'vlBest_acc_78_model_epoch_0180.pyth')
        mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
    elif dataset == 'PATHMNIST_REVERSE':
        DATA_DIR = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/pathmnist_random_selection_wt_imagenet/logs/'
        DATA_DIR_ACTIVE = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/pathmnist_active_selection_wt_imagenet/logs/'
        # checkpoint_file = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/PATHMNIST_REVERSE/resnet18/trial1/episode_0/vlBest_acc_94_model_epoch_0200.pyth'
        checkpoint_file = os.path.join(df_save_dir, 'vlBest_acc_94_model_epoch_0200.pyth')
        mean, std = [.5, .5, .5], [.5, .5, .5]

    # gpu set to False: use calculated features and active learning method score
    # gpu = True
    gpu = False

    # Active learning selection strategy
    al = 'uncertainty'
    # al = 'coreset'
    # al = 'margin'
    # al = 'bald'
    # al = 'consistency'
    # al = 'vaal'

    # Load dataset
    data_obj = Data(cfg)
    model = load_model(cfg, checkpoint_file)
    train_data, test_size = load_train_data(cfg, data_obj)
    test_data, test_size = load_test_data(cfg, data_obj)
    test_data.transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(std=std, mean=mean),
    ])
    if al is 'consistency':
        train_data.transform = transforms.Compose([
            transforms.RandomCrop(size=[32, 32], padding=4),
            transforms.ToTensor(),
            transforms.Normalize(std=std, mean=mean),
        ])
    else:
        train_data.transform = test_data.transform

    # Create df
    if os.path.isfile(os.path.join(df_save_dir, dataset + '.pkl')):
        full_df = pd.read_pickle(os.path.join(df_save_dir, dataset + '.pkl'))
        full_df = full_df.dropna()  # need to dropna before experiments all done
    else:
        file_info_random, file_info_active = read_dir(DATA_DIR), read_dir_by_acquisition(DATA_DIR_ACTIVE)
        file_info = dict(list(file_info_random.items()) + list(file_info_active.items()))
        full_df = pd.DataFrame.from_dict(file_info, orient='index')
        full_df = full_df.astype({'auc': 'float64', 'ratio': 'float64', 'trial': 'float64'})
        full_df.to_pickle(os.path.join(df_save_dir, dataset + '.pkl'))
        full_df = full_df.dropna()
    print(full_df.dtypes)

    # # get coreset index, for debugging
    # budget = len(test_data)
    # _ = get_coreset_index(cfg, data_obj, model, test_data, budget=budget)

    # Calculate score
    # calculate_score = False
    # if calculate_score:
    #     if al is 'coreset':
    #         features = coreset(cfg, data_obj, model, test_data)
    #         full_df = full_df[full_df['ratio'] <= 0.005]
    #         full_df['local_score'] = full_df['indices'].parallel_apply(calculate_coreset_score) / len(
    #             full_df['indices'])  # reduce score by average
    #     else:
    #         score = calculate_global_score(al, cfg, data_obj, model, test_data)
    #
    #         sns.displot(x=score)
    #         plt.savefig(os.path.join(plot_save_dir, dataset, al + '_score.png'))
    #
    #         full_df['local_score'] = full_df['indices'].apply(calculate_local_score) / len(
    #             full_df['indices'])  # reduce score by average

    # df = full_df[full_df['acquisition'] == 'random']
    df = full_df

    rewrite_df = True
    if os.path.isfile(os.path.join(df_save_dir, dataset + '_full_info.pkl')) and not rewrite_df:
        df = pd.read_pickle(os.path.join(df_save_dir, dataset + '_full_info.pkl'))
    else:
        calculate_score = True
        if calculate_score:
            if al is 'coreset':
                features = coreset(cfg, data_obj, model, test_data)
                df = df[df['ratio'] <= 0.005]
                df['local_score'] = df['indices'].parallel_apply(calculate_coreset_score) / len(
                    df['indices'])  # reduce score by average
            else:
                score = calculate_global_score(al, cfg, data_obj, model, test_data)

                sns.displot(x=score)
                plt.savefig(os.path.join(plot_save_dir, dataset, al + '_score.png'))

                df['local_score'] = df['indices'].apply(calculate_local_score) / len(df['indices'])

        calculate_difficulty = True
        if calculate_difficulty:
            # df = df[df['ratio'] <= 0.05]  # for hard count correlation analysis
            train_dy_metrics = pd.read_json(
                os.path.join('/media/ntu/volume2/home/s121md302_06/workspace/code/cartography/dynamics_logs/',
                             train_data.__class__.__name__.upper(),
                             'training_td_metrics.jsonl'), lines=True)
            train_dy_metrics = train_dy_metrics.assign(corr_frac=lambda d: d.correctness / d.correctness.max())
            train_dy_metrics = train_dy_metrics.set_index('guid')
            conf_outer_list = []
            corr_outer_list = []
            label_outer_list = []
            conf_count_easy = []
            conf_count_medium = []
            conf_count_hard = []
            variability_score = []
            confidence_score = []

            full_label_list = [test_data[i][1] for i in range(len(test_data))]

            for indice_list in tqdm(df['indices'], desc="Adding Labels"):
                conf_list = []
                corr_list = []
                var_list = []
                label_list = []
                for index in indice_list:
                    try:
                        # confidence, correctness, variability = train_dy_metrics.loc[index]['confidence'], \
                        #                                        train_dy_metrics.loc[index]['corr_frac'], \
                        #                                        train_dy_metrics.loc[index]['variability']
                        # variability = train_dy_metrics.loc[index]['variability']
                        # confidence = train_dy_metrics.loc[index]['confidence']
                        label = full_label_list[index]
                    except:
                        print('train_dy_metrics error', index)
                    # conf_list.append(confidence)
                    # corr_list.append(correctness)
                    # var_list.append(variability)
                    label_list.append(label)
                # conf_list, corr_list = np.array(conf_list), np.array(corr_list)

                # bins = [0.0, 0.33, 0.67, 1.0]
                #
                # conf_count = np.histogram(conf_list, bins)[0]
                # conf_count_easy.append(conf_count[2] / len(conf_list))
                # conf_count_medium.append(conf_count[1] / len(conf_list))
                # conf_count_hard.append(conf_count[0] / len(conf_list))
                # variability_score.append(sum(var_list) / len(var_list))
                # confidence_score.append(sum(conf_list) / len(conf_list))
                # conf_outer_list.append(conf_list)
                # corr_outer_list.append(corr_list)
                label_outer_list.append(label_list)

            # df['conf_count_easy'] = conf_count_easy
            # df['conf_count_medium'] = conf_count_medium
            # df['conf_count_hard'] = conf_count_hard
            # df['variability_score'] = variability_score
            # df['confidence_score'] = confidence_score
            df['labels'] = label_outer_list
        df.to_pickle(os.path.join(df_save_dir, dataset + '_full_info.pkl'))

    # Plot class distribution
    plot_distribution_histogram = False
    if plot_distribution_histogram:
        index_to_class_dict = train_data.info['label']
        uniform_label = list(range(10)) * 100
        plot_class_dist(uniform_label, index_to_class_dict=index_to_class_dict, ylim=(None, 200))
        # ratio = 0.01  # small ratio
        # ratio = 0.20  # small ratio for vaal only
        ratio = 0.5  # large ratio
        random_label = df[(df['acquisition'] == 'random') & (df['ratio'] == ratio) & (df['trial'] == 1)]['labels'][0]
        plot_class_dist(random_label, index_to_class_dict=index_to_class_dict, ratio=ratio)

        for acquisition in ['uncertainty', 'coreset', 'margin', 'bald', 'consistency', 'vaal']:
            active_indices = \
            full_df[(full_df['acquisition'] == acquisition) & (full_df['ratio'] == ratio) & (full_df['trial'] == 1)][
                'indices']
            label_list = []
            for indices_list in active_indices:
                label_list.append([test_data[index][1] for index in indices_list])
            active_label = label_list
            plot_class_dist(active_label, index_to_class_dict=index_to_class_dict, acquisition=acquisition, ratio=ratio)

    # Correlation analysis
    ratios = np.sort(df['ratio'].unique())
    corr_list, p_value_list = [], []
    print('{} \t {} \t {}'.format('ratio', 'corr', 'p_value'))

    # for ratio in ratios:
    for ratio in ratios[::-1]:
        if al is 'coreset' and ratio > 0.005:  # limit calculation complexity
            continue
        ratio_df = df[(df['ratio'] == ratio) &
                      (df['acquisition'] == 'random') &
                      (df['auc'].notna())]

        # corrletion of active learning method local_score
        corr, p_value = scipy.stats.pearsonr(ratio_df['auc'], ratio_df['local_score'])
        corr_list.append(corr)
        p_value_list.append(p_value)
        print('{} \t {:.4f} \t {:.4f}'.format(ratio, corr, p_value))

        # Local score histogram plot
        plot_local_score_histogram = False
        if plot_local_score_histogram:
            sns.displot(x=ratio_df['local_score'])
            plt.title(al + str(ratio))
            Path(os.path.join(plot_save_dir, dataset, al)).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(plot_save_dir, dataset, al, str(ratio) + '_histogram.png'))
            plt.show()

        # Plot local score scatter plot
        plot_local_score = True
        if plot_local_score:
            fontsize = 30
            markersize = 40
            linewidth = 4
            plt.rcParams.update({'font.size': fontsize})

            # create scatter plot
            active_ratio_df = df[(df['ratio'] == ratio) &
                                 (df['acquisition'] == al) &
                                 (df['auc'].notna())]

            corr, p_value = scipy.stats.pearsonr(ratio_df['auc'], ratio_df['local_score'])

            # Specify xlim
            ax_xlim_low = ratio_df['local_score'].min() - 0.05 * (
                    ratio_df['local_score'].max() - ratio_df['local_score'].min())
            ax_xlim_high = ratio_df['local_score'].max() + 0.05 * (
                    ratio_df['local_score'].max() - ratio_df['local_score'].min())
            ax2_xlim_low = active_ratio_df['local_score'].min() - 0.01 * (
                    ratio_df['local_score'].max() - ratio_df['local_score'].min())
            ax2_xlim_high = active_ratio_df['local_score'].max() + - 0.01 * (
                    ratio_df['local_score'].max() - ratio_df['local_score'].min())

            ratio_df = pd.concat([ratio_df, active_ratio_df])
            fig = plt.figure(figsize=[20, 10])
            ax = fig.add_subplot(121)
            sns.regplot(ratio_df['local_score'], ratio_df['auc'], color='grey', label='random',
                        scatter_kws={'s': markersize})
            ax.set_ylim(None, 1)
            ax2 = fig.add_subplot(122, sharey=ax)
            sns.scatterplot(active_ratio_df['local_score'], active_ratio_df['auc'], color='red', label=al,
                            s=2 * markersize)

            ax.set_xlim(ax_xlim_low, ax_xlim_high)
            ax2.set_xlim(ax2_xlim_low, ax2_xlim_high)

            # To specify the number of ticks on both or any single axes
            plt.locator_params(axis='y', nbins=4)
            plt.locator_params(axis='x', nbins=4)
            ax.locator_params(axis='x', nbins=4)

            # hide the spines between ax and ax2
            ax2.yaxis.tick_right()
            ax.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            # ax2.spines['right'].set_visible(False)
            ax.yaxis.tick_left()
            # hide xlabel
            ax.set_xlabel(None)
            ax2.set_xlabel(None)
            # hide and ax2 yaxis
            plt.gca().axes.get_yaxis().set_visible(False)
            # remove legend
            ax2.get_legend().remove()

            d = .015  # how big to make the diagonal lines in axes coordinates
            # arguments to pass plot, just so we don't keep repeating them
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax2.plot((-d, +d), (-d, +d), **kwargs)

            ax.set_ylabel('AUC')
            ax.plot([], [], ' ', label='\u03C1 = ' + "{:.2f}".format(corr) + '\np-value = ' + "{:.2f}".format(p_value))

            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            lines = lines_1 + lines_2
            labels = labels_1 + labels_2
            ax.legend(lines, labels, loc='lower left', )

            fig.tight_layout()
            Path(os.path.join(plot_save_dir, dataset, al)).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(plot_save_dir, dataset, al, str(ratio) + '.png'))
            # plt.show()

            # plot random correlation only
            fig = plt.figure(figsize=[11.33, 15.24])
            ax = fig.add_subplot(111)
            sns.regplot(ratio_df['local_score'], ratio_df['auc'], color='grey', label='random',
                        scatter_kws={'s': markersize})
            ax.set_ylim(None, 1)
            ax.set_xlim(ax_xlim_low, ax_xlim_high)
            plt.locator_params(axis='y', nbins=4)
            plt.locator_params(axis='x', nbins=4)
            ax.locator_params(axis='x', nbins=4)
            ax.set_xlabel(None)
            ax.set_ylabel('AUC')
            ax.plot([], [], ' ',
                    label='\u03C1 = ' + "{:.2f}".format(corr) + '\np-value = ' + "{:.2f}\n".format(p_value))
            ax.legend(loc='lower left',
                      # fontsize=fontsize
                      )
            fig.tight_layout()
            Path(os.path.join(plot_save_dir, dataset, al)).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(plot_save_dir, dataset, al, str(ratio) + '_random_only.png'))
            # plt.show()

        # Correlation
        # sns.regplot(ratio_df['conf_count_hard']+ratio_df['conf_count_medium'], ratio_df['auc'], color='grey', label='random')
        # plt.title(al + '_' + str(ratio) + '_' + 'hard+medium(confidence)')
        # plt.savefig(os.path.join(plot_save_dir, dataset, al, str(ratio) + '_' + 'hard+medium(confidence)' + '.png'))
        # plt.show()

    # Plot correlation summary
    plot_correlation_summary = False
    if plot_correlation_summary:
        sns.lineplot(x=ratios, y=corr_list)
        sns.lineplot(x=ratios, y=p_value_list)
        plt.title(al)
        plt.show()
        plt.savefig(os.path.join(plot_save_dir, dataset, 'summary.png'))

    # Print correlation summary
    for idx, (ratio, corr, p_value) in enumerate(zip(ratios, corr_list, p_value_list)):
        print('{} \t {:.4f} \t {:.4f}'.format(ratio, corr, p_value))
