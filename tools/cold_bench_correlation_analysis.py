import argparse
import glob
import itertools
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import torch
import torch.nn as nn
from pandarallel import pandarallel
from scipy import spatial
import sklearn
from torchvision import transforms
from tqdm import tqdm
import mmcv

import pycls.core.builders as model_builder
import pycls.utils.checkpoint as cu
from pycls.core.config import cfg
from pycls.datasets.data import Data

pandarallel.initialize()


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning Cold Start Benchmark')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    return parser


# Plot class distribution (have trouble)
def plot_distribution_histogram(ratio=0.5):
    uniform_label = list(range(10)) * 100
    plot_class_dist(uniform_label, ylim=(None, 200))
    random_label = df[(df['acquisition'] == 'Random') & (df['ratio'] == ratio) & (df['trial'] == 1)]['labels'][0]
    plot_class_dist(random_label, ratio=ratio)

    for acquisition in ['uncertainty', 'coreset', 'margin', 'bald', 'consistency', 'vaal']:
        active_indices = \
            full_df[(full_df['acquisition'] == acquisition) & (full_df['ratio'] == ratio) & (full_df['trial'] == 1)][
                'indices']
        label_list = []
        for indices_list in active_indices:
            label_list.append([test_data[index][1] for index in indices_list])
        active_label = label_list
        plot_class_dist(active_label, acquisition=acquisition, ratio=ratio)


def plot_class_dist(labels, acquisition='Random', ratio=0.01, title=None, figsize=[40, 40], ylim=(None, None)):
    classes, counts = np.unique(labels, return_counts=True)
    df_rel = pd.DataFrame(columns=['classes', 'counts'])
    df_rel['classes'], df_rel['counts'] = classes, counts
    df_rel = df_rel.sort_values('classes')
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.autolayout"] = True
    df_rel.plot(x='classes', y='counts', kind='barh', stacked=True,
                title=title,
                legend=None,
                figsize=figsize, colormap='Reds_r',
                xlabel=None,
                xticks=None,
                ylabel=None,
                yticks=None,
                # width=0.9
                )
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.ylim(ylim)
    save_path = os.path.join(plot_save_dir, dataset, acquisition + '_' + str(ratio) + '_distribution_histogram.png')
    print(save_path)
    plt.savefig(save_path)
    plt.show()


def read_dir_by_acquisition(path, dataset='PathMNIST'):
    file_list = glob.glob(path + '*.out')
    file_info = {}
    acquisition_dict = {
        'uncertainty': 'Entropy',
        'bald': 'BALD',
        'consistency': 'Consistency',
        'coreset': 'Coreset',
        'margin': 'Margin',
        'vaal': 'VAAL'
    }
    for file in file_list:
        file_info[file] = {'trial': file.split('-')[1],
                           'ratio': file.split('-p')[-1].split('-')[0],
                           'acquisition': acquisition_dict[file.split('-')[-1].split('.out')[0]],
                           }

        f = open(file, "r")
        lines = f.readlines()

        al_sorted_idx_file = os.path.join(df_save_dir, dataset,
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
                           'acquisition': 'Random'
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


def calculate_global_score(al, cfg, dataObj, train_data, model, dataset, recalculate_idx=False):
    if al == 'Entropy':
        score = entropy(cfg, dataObj=dataObj, train_data=train_data, model=model, dataset=dataset, recalculate_idx=recalculate_idx)
    elif al == 'Coreset':
        raise NotImplementedError('Use coreset score calculation.')
    elif al == 'Margin':
        score = margin(cfg, dataObj=dataObj, train_data=train_data, model=model, dataset=dataset, recalculate_idx=recalculate_idx)
    elif al == 'BALD':
        score = bald(cfg, dataObj=dataObj, train_data=train_data, model=model, dataset=dataset, recalculate_idx=recalculate_idx)
    elif al == 'Consistency':
        score = consistency(cfg, dataObj=dataObj, train_data=train_data, model=model, dataset=dataset, recalculate_idx=recalculate_idx)
    elif al == 'VAAL':
        score = vaal(cfg, dataObj=dataObj, train_data=train_data, model=model, dataset=dataset, recalculate_idx=recalculate_idx)
    else:
        raise NotImplementedError(al)
    return score


def entropy(cfg, dataObj, train_data, model, dataset, recalculate_idx=False):
    """
    Implements the uncertainty principle as a acquisition function.
    """
    if recalculate_idx:
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


def consistency(cfg, dataObj, train_data, model, dataset, recalculate_idx=False):
    if recalculate_idx:
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


def margin(cfg, dataObj, train_data, model, dataset, recalculate_idx=False):
    """
    Implements the uncertainty principle as a acquisition function.
    """
    if recalculate_idx:
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


def vaal(cfg, dataObj, train_data, model, dataset, recalculate_idx=False):
    if recalculate_idx:
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


def bald(cfg, dataObj, train_data, model, dataset, recalculate_idx=False):
    if recalculate_idx:
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


def coreset(cfg, dataObj, model, dataset, recalculate_idx=False):
    if recalculate_idx:
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


def get_coreset_index(cfg, dataObj, train_data, model, dataset, budget):
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


def plot_local_score_auc_random_plus_active(ratio_df, al, df, ratio, plot_save_dir, dataset,
                                            fontsize=35,
                                            markersize=150,
                                            linewidth=4,
                                            figsize=(18, 9),
                                            show=False,
                                            ):
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['axes.linewidth'] = linewidth
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    fig = plt.figure(figsize=figsize)

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

    # Specify ylim
    random_low = 1 - (1 - ratio_df['auc'].mean()) * 3
    active_low = 1 - (1 - active_ratio_df['auc'].min()) * 1.1
    ax_ylim_high = 1.0
    ax_ylim_low = min(random_low, active_low)
    # ax2_ylim_low = 1 - (1 - ratio_df['auc'].mean()) * 3
    # ax2_ylim_high = 1.0

    ratio_df = pd.concat([ratio_df, active_ratio_df])

    ax = fig.add_subplot(121)
    sns.regplot(ratio_df['local_score'], ratio_df['auc'], color='grey', label='Random',
                scatter_kws={'s': markersize}, line_kws={'linewidth': linewidth * 1.5})
    ax.set_ylim(None, 1)
    ax2 = fig.add_subplot(122, sharey=ax)
    sns.scatterplot(active_ratio_df['local_score'], active_ratio_df['auc'], color='red', label=al,
                    s=markersize)

    ax.set_xlim(ax_xlim_low, ax_xlim_high)
    ax2.set_xlim(ax2_xlim_low, ax2_xlim_high)
    ax.set_ylim(ax_ylim_low, ax_ylim_high)
    # ax2.set_ylim(ax2_ylim_low, ax2_ylim_high)

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
    # ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs) # top left diagonal marker

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs) # top right diagonal marker
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    ax.set_ylabel('AUC')
    # ax.plot([], [], ' ', label='\u03C1 = ' + "{:.2f}".format(corr) + '\np-value = ' + "{:.2f}".format(p_value))

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax.legend(lines, labels, loc='lower left', )

    fig.tight_layout()
    Path(os.path.join(plot_save_dir, dataset, al)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(plot_save_dir, dataset, al, str(ratio) + '.png'),
                bbox_inches='tight', transparent=True,
                )
    if show:
        plt.show()


def plot_local_score_auc_random(ratio_df, al, df, ratio, plot_save_dir, dataset,
                                fontsize=35, markersize=150, linewidth=4, figsize=(4.46 * 2, 6 * 2),
                                show=False,
                                ):
    # Specify xlim
    ax_xlim_low = ratio_df['local_score'].min() - 0.05 * (
            ratio_df['local_score'].max() - ratio_df['local_score'].min())
    ax_xlim_high = ratio_df['local_score'].max() + 0.05 * (
            ratio_df['local_score'].max() - ratio_df['local_score'].min())

    # Specify ylim
    ax_ylim_low = 1 - (1 - ratio_df['auc'].mean()) * 3
    ax_ylim_high = 1.0

    corr, p_value = scipy.stats.pearsonr(ratio_df['auc'], ratio_df['local_score'])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    sns.regplot(ratio_df['local_score'], ratio_df['auc'], color='grey', label='Random',
                scatter_kws={'s': markersize}, line_kws={'linewidth': linewidth * 1.5})
    ax.set_ylim(ax_ylim_low, ax_ylim_high)
    ax.set_xlim(ax_xlim_low, ax_xlim_high)
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=2)
    ax.locator_params(axis='x', nbins=2)
    ax.set_xlabel(None)
    ax.set_ylabel('AUC')
    ax.plot([], [], ' ',
            label='\u03C1 = ' + "{:.2f}".format(corr) + '\np-value = ' + "{:.2f}\n".format(p_value))
    ax.legend(loc='lower right',
              # fontsize=fontsize
              )
    fig.tight_layout()
    Path(os.path.join(plot_save_dir, dataset, al)).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(plot_save_dir, dataset, al, str(ratio) + '_random_only.png'),
                bbox_inches='tight', transparent=True,
                )
    if show:
        plt.show()


def load_info_to_df(al, checkpoint_file, dataset, df_save_dir, mean, std,
                    rewrite_basic_df=False, rewrite_full_info_df=False):
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
    if al.lower() == 'consistency':
        train_data.transform = transforms.Compose([
            transforms.RandomCrop(size=[32, 32], padding=4),
            transforms.ToTensor(),
            transforms.Normalize(std=std, mean=mean),
        ])
    else:
        train_data.transform = test_data.transform

    # Create basic df
    if os.path.isfile(os.path.join(df_save_dir, dataset + '.pkl')) and not rewrite_basic_df:
        df = pd.read_pickle(os.path.join(df_save_dir, dataset + '.pkl'))
        df = df.dropna()  # need to dropna before experiments all done
    else:
        file_info_random = read_dir(DATA_DIR)
        if os.path.exists(DATA_DIR_ACTIVE):
            file_info_active = read_dir_by_acquisition(DATA_DIR_ACTIVE, dataset=train_data.__class__.__name__)
            file_info = dict(list(file_info_random.items()) + list(file_info_active.items()))
        else:
            file_info = file_info_random
        df = pd.DataFrame.from_dict(file_info, orient='index')
        df = df.astype({'auc': 'float64', 'ratio': 'float64', 'trial': 'float64'})
        df.to_pickle(os.path.join(df_save_dir, dataset + '.pkl'))
        df = df.dropna()
    print(df.dtypes)

    if os.path.isfile(os.path.join(df_save_dir, dataset + '_full_info.pkl')) and not rewrite_full_info_df:
        df = pd.read_pickle(os.path.join(df_save_dir, dataset + '_full_info.pkl'))
    else:
        calculate_score = True
        if calculate_score:
            if al.lower() == 'coreset':
                get_coreset_index(cfg, dataObj=data_obj, train_data=train_data, model=model, dataset=test_data,
                                          budget=len(train_data))  # debug only
                features = coreset(cfg, data_obj, model, test_data)
                df = df[df['ratio'] <= 0.005]
                df['local_score'] = df['indices'].parallel_apply(calculate_coreset_score) / len(
                    df['indices'])  # reduce score by average
            else:
                score = calculate_global_score(al, cfg, data_obj, train_data, model, test_data, recalculate_idx=True)

                sns.displot(x=score)
                plt.savefig(os.path.join(plot_save_dir, dataset, al + '_score.png'))

                local_score = []
                for i, indices in enumerate(tqdm(df.to_dict()['indices'].values(), desc="Calculate local score")):
                    local_score.append(np.sum([score[index] for index in indices]))

                df['local_score'] = np.array(local_score) / len(df['indices'])

        calculate_difficulty = False
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

            full_label_list = [test_data[i][-1] for i in range(len(test_data))]

            for indice_list in tqdm(df['indices']):
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

    return df, train_data, test_data


def scatter_and_multi_barplot_result(x1, y1, sd1, m1=[], lb1='random',
                                     x2_list=[], y2_list=[], sd2_list=[], m2_list=[], lb2_list=[],
                                     xlabel='', ylabel='', title='',
                                     xmin=None, xmax=None, xticks=None,
                                     ymin=0.88, ymax=1.0, yticks=[0.88, 0.92, 0.96, 1.0],
                                     xlog=False, upper=False, alpha=0.3, legend=True,
                                     markersize=30, elinewidth=10, linewidth=10, fontsize=120, figsize=(50, 40),
                                     gray_color=[92 / 255, 102 / 255, 112 / 255],
                                     red_color=[208 / 255, 53 / 255, 48 / 255],
                                     ROOT=None,
                                     save_dir=None,
                                     ):
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['axes.linewidth'] = linewidth
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    assert len(x2_list) == len(y2_list) == len(sd2_list) == len(m2_list) == len(lb2_list)

    marker_list = ['v', '^', '<', '>', 's', 'd', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', '|', '_']

    plt.scatter(x1, y1,
                color=gray_color,
                s=markersize * 30, label=lb1)

    if len(x2_list) > 0:
        for i, (x2, y2, sd2, m2, lb2) in enumerate(zip(x2_list, y2_list, sd2_list, m2_list, lb2_list)):
            plt.errorbar(x2, y2, yerr=sd2, fmt=marker_list[i], color=red_color,
                         markersize=markersize * 1.5,
                         alpha=alpha,
                         ecolor=red_color, elinewidth=elinewidth, capsize=0, label=lb2)

    if len(m1) > 0 and upper:
        plt.plot(x1, m1, color='lightgray', linewidth=linewidth)
    if len(m2_list) > 0 and upper:
        for i, (x2, m2) in enumerate(zip(x2_list, m2_list)):
            plt.plot(x2, m2, color=red_color, linewidth=linewidth)

    if legend:
        plt.legend(loc='lower right')

    if xlog:
        ax.set_xscale('log')
    plt.grid(axis='y', alpha=0.5, linewidth=linewidth)
    if xlabel != '':
        plt.xlabel(xlabel)
    if ylabel != '':
        plt.ylabel(ylabel)
    if xmin is not None and xmax is not None:
        plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax);
    plt.yticks(yticks)
    if title != '':
        plt.title(title.replace("_", " "))
    plt.show()
    if title != '':
        fig.savefig(os.path.join(save_dir, title+'_multiple_active_'+ylabel.replace(" ", "_")+'.png'),
                    bbox_inches='tight', pad_inches=0.05, dpi=200)


def load_logs(runs=[], plist=[], suffix=None, data='mnist10', zfill=5, ROOT=None):
    p = [str(i).zfill(zfill) for i in plist]
    auc = np.zeros((len(runs), len(p)), dtype='float')
    for i, run in enumerate(tqdm(runs)):
        for j, percentage in enumerate(p):
            try:
                out = os.path.join(ROOT, suffix, 'logs', data + '-' + str(run) + '-p0.' + percentage + '.out')
                text_file = open(out, 'r')
                lines = text_file.read().split('\n')
                auc_info = [line for line in lines if 'AVERAGE | AUC =' in line]
                auc[i, j] = float(auc_info[0][15:])


            except:
                print('ERROR in {}'.format(os.path.join(suffix,
                                                        'logs',
                                                        data + '-' + str(run) + '-p0.' + percentage + '.out')))
                raise

    return auc


def load_logs_by_acquisition(runs=[], plist=[],
                             suffix=None, data='mnist10', zfill=5, ROOT=None,
                             acquisition_function='entropy'):
    p = [str(i).zfill(zfill) for i in plist]
    auc = np.zeros((len(runs), len(p)), dtype='float')
    acquisition_function = 'uncertainty' if acquisition_function == 'entropy' else acquisition_function
    for i, run in enumerate(tqdm(runs)):
        for j, percentage in enumerate(p):
            out = os.path.join(ROOT, suffix, 'logs',
                               data + '-' + str(run) + '-p0.' + percentage + '-' + acquisition_function + '.out')
            text_file = open(out, 'r')
            lines = text_file.read().split('\n')
            error_message = [line for line in lines if 'Cannot cover all classes' in line]
            if len(error_message) > 0:
                continue
            auc_info = [line for line in lines if 'AVERAGE | AUC =' in line]
            if len(auc_info) == 0:
                print('ERROR in {}'.format(
                    data + '-' + str(run) + '-p0.' + percentage + '-' + acquisition_function + '.out'))
                continue
            auc[i, j] = float(auc_info[0][15:])

    return auc

def plot_multiple_random_scatter_active_selection(num_run_random, num_run_active,
                                                  num_train,
                                                  plist,
                                                  data='pathmnist',
                                                  xlabel='Number of images', ylabel='AUC',
                                                  title='',
                                                  #                               flag_list=['uncertainty'],
                                                  flag_list=[],
                                                  extend_ratio = 1.1,
                                                  xmin=None, xmax=None,
                                                  xticks=None,
                                                  ymin=0.7, ymax=1.0,
                                                  yticks=[0.7, 0.8, 0.9, 1.0],
                                                  alpha=0.3,
                                                  legend=True,
                                                  xlog=True,
                                                  ROOT=None,
                                                  save_dir=None,
                                                  upper=False,
                                                  markersize=30, elinewidth=10,
                                                  linewidth=10, fontsize=120, figsize=(50, 40),
                                                  ):
    random_auc = load_logs(runs=[i + 1 for i in range(num_run_random)],
                           plist=plist,
                           suffix=data + '_random_selection_wt_imagenet',
                           data=data,
                           zfill=5,
                           ROOT=ROOT,
                           )

    active_auc_list = [load_logs_by_acquisition(runs=[i + 1 for i in range(num_run_active)],
                                                plist=plist,
                                                suffix=data + '_active_selection_wt_imagenet',
                                                data=data,
                                                zfill=5,
                                                ROOT=ROOT,
                                                acquisition_function=flag.lower(),
                                                ) for flag in flag_list]

    scatter_and_multi_barplot_result(
        x1=np.tile(np.array([num_train * i / 100000.0 for i in plist]), (num_run_random, 1)),
        y1=random_auc,
        sd1=np.std(random_auc, axis=0),
        m1=np.max(random_auc, axis=0),
        lb1='Random',

        # x2_list=[[num_train * i / 100000.0 for i in plist]] * len(flag_list),
        x2_list=[[(num_train * i * extend_ratio) / 100000.0 for i in plist]] * len(flag_list),
        y2_list=[np.mean(active_auc, axis=0) for active_auc in active_auc_list],
        sd2_list=[np.std(active_auc, axis=0) for active_auc in active_auc_list],
        m2_list=[np.max(active_auc, axis=0) for active_auc in active_auc_list],
        lb2_list=flag_list,

        alpha=alpha,
        xlog=xlog,
        xmin=xmin, xmax=xmax, xticks=xticks,
        ymin=ymin, ymax=ymax, yticks=yticks,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        legend=legend,
        markersize=markersize,
        elinewidth=elinewidth,
        linewidth=linewidth,
        fontsize=fontsize,
        figsize=figsize,
        ROOT=ROOT,
        save_dir=save_dir,
        upper=upper,
        )

    return random_auc, active_auc_list

def plot_split_class_distribution(df_dir=None, split='easy'):
    all_indices = np.load(os.path.join(df_dir, split + '_sorted_idx.npy'))
    # split_indices = all_indices[:len(all_indices) // 3]
    split_indices = all_indices[:len(all_indices) // 100]
    label_list = [test_data[index][-1] for index in split_indices]
    plot_class_dist(label_list, title=split)


if __name__ == '__main__':
    cfg.merge_from_file(argparser().parse_args().cfg_file)
    dataset = cfg.DATASET.NAME

    # Define DATA_DIR: logs dir
    # checkpoint_file: model checkpoint
    # plot_save_dir: dir to save plots
    # df_save_dir: dir to save df
    ROOT = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/'
    plot_save_dir = os.path.join(ROOT, 'plot')
    df_save_dir = os.path.join(ROOT, 'df')
    mmcv.mkdir_or_exist(plot_save_dir)
    mmcv.mkdir_or_exist(df_save_dir)
    # plot 1d scatter plot, active and random
    ROOT = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/'
    num_run_random, num_run_active, num_train = 20, 2, 89996
    plist = [15, 100, 1000, 10000]
    flag_list = ['Consistency', 'VAAL',  'Margin', 'Entropy', 'Coreset', 'BALD', ]
    random_auc, active_auc = plot_multiple_random_scatter_active_selection(num_run_random=num_run_random,
                                                                           num_run_active=num_run_active,
                                                                           num_train=num_train,
                                                                           plist=plist,
                                                                           data='pathmnist',
                                                                           title='PathMNIST',
                                                                           ymin=0.65, ymax=1.0,
                                                                           yticks=[0.7, 0.8, 0.9, 1.0],
                                                                           xlog=True,
                                                                           alpha=0.5,
                                                                           flag_list=flag_list,
                                                                           ROOT=ROOT,
                                                                           save_dir=plot_save_dir,
                                                                           upper=False,
                                                                           markersize=30,
                                                                           elinewidth=10,
                                                                           linewidth=10,
                                                                           fontsize=120,
                                                                           figsize=(50, 40),
                                                                           )

    if dataset == 'CIFAR10_REVERSE':
        DATA_DIR = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/cifar10_random_selection_wt_imagenet/logs/'
        DATA_DIR_ACTIVE = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/cifar10_active_selection_wt_imagenet/logs/'
        # checkpoint_file = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/CIFAR10_REVERSE/resnet18/reverse_trial1/episode_0/vlBest_acc_78_model_epoch_0180.pyth'
        checkpoint_file = os.path.join(df_save_dir, 'vlBest_acc_78_model_epoch_0180.pyth')
        mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
    elif dataset == 'IMBALANCED_CIFAR10_REVERSE':
        DATA_DIR = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/cifar10_random_selection_wt_imagenet/logs/'
        DATA_DIR_ACTIVE = 'None'
        # checkpoint_file = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/CIFAR10_REVERSE/resnet18/reverse_trial1/episode_0/vlBest_acc_78_model_epoch_0180.pyth'
        checkpoint_file = os.path.join(df_save_dir, 'vlBest_acc_77_model_epoch_0180.pyth')
        mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
    elif dataset == 'PATHMNIST_REVERSE':
        DATA_DIR = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/pathmnist_random_selection_wt_imagenet/logs/'
        DATA_DIR_ACTIVE = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/pathmnist_active_selection_wt_imagenet/logs/'
        # checkpoint_file = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/PATHMNIST_REVERSE/resnet18/trial1/episode_0/vlBest_acc_94_model_epoch_0200.pyth'
        checkpoint_file = os.path.join(df_save_dir, 'vlBest_acc_94_model_epoch_0200.pyth')
        mean, std = [.5, .5, .5], [.5, .5, .5]
    elif dataset == 'DERMAMNIST_REVERSE':
        DATA_DIR = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/dermamnist_random_selection_wt_imagenet/logs/'
        DATA_DIR_ACTIVE = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/dermamnist_active_selection_wt_imagenet/logs/'
        # checkpoint_file = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/PATHMNIST_REVERSE/resnet18/trial1/episode_0/vlBest_acc_94_model_epoch_0200.pyth'
        checkpoint_file = os.path.join(df_save_dir, 'vlBest_acc_94_model_epoch_0200.pyth')
        mean, std = [.5, .5, .5], [.5, .5, .5]
    elif dataset == 'BLOODMNIST_REVERSE':
        DATA_DIR = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/bloodmnist_random_selection_wt_imagenet/logs/'
        DATA_DIR_ACTIVE = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/bloodmnist_active_selection_wt_imagenet/logs/'
        # checkpoint_file = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/PATHMNIST_REVERSE/resnet18/trial1/episode_0/vlBest_acc_94_model_epoch_0200.pyth'
        checkpoint_file = os.path.join(df_save_dir, 'vlBest_acc_93_model_epoch_0180.pyth')
        mean, std = [.5, .5, .5], [.5, .5, .5]
    elif dataset == 'ORGANAMNIST_REVERSE':
        DATA_DIR = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/organamnist_random_selection_wt_imagenet/logs/logs/'
        DATA_DIR_ACTIVE = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/organamnist_active_selection_wt_imagenet/logs/'
        # checkpoint_file = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/PATHMNIST_REVERSE/resnet18/trial1/episode_0/vlBest_acc_94_model_epoch_0200.pyth'
        checkpoint_file = os.path.join(df_save_dir, 'vlBest_acc_96_model_epoch_0040.pyth')
        mean, std = [.5], [.5]



    # Active learning selection strategy
    # for al in ['Entropy', 'BALD', 'Consistency', 'Margin', 'VAAL', 'Coreset']:
    for al in ['Coreset']:
    # for al in ['Coreset']:
    # for al in ['VAAL']:
        df, train_data, test_data = load_info_to_df(al=al, dataset=dataset,
                                                    checkpoint_file=checkpoint_file,
                                                    df_save_dir=df_save_dir,
                                                    mean=mean, std=std,
                                                    rewrite_basic_df=True, rewrite_full_info_df=True
                                                    )

        # continue # derma & blood debug only

        ratios = np.sort(df['ratio'].unique())
        # ratios = [ratios[60], ratios[-4]]
        ratios = [0.20, 0.5, 0.6, 0.01]
        print(len(ratios), ratios)

        # for split in ['easy', 'ambiguous', 'hard']:
        #     plot_split_class_distribution(df_dir=os.path.join(df_save_dir, train_data.__class__.__name__), split=split)

        corr_list, p_value_list = [], []
        print('{} \t {} \t {}'.format('ratio', 'corr', 'p_value'))

        # Plot correlation
        plot_correlation = False
        if plot_correlation:
            for ratio in ratios:
                # for ratio in ratios[::-1]:
                if al == 'Coreset' and ratio > 0.005:  # limit calculation complexity
                    continue
                ratio_df = df[(df['ratio'] == ratio) &
                              (df['acquisition'] == 'Random') &
                              (df['auc'].notna())]

                # corrletion of active learning method local_score
                corr, p_value = scipy.stats.pearsonr(ratio_df['auc'], ratio_df['local_score'])
                corr_list.append(corr)
                p_value_list.append(p_value)
                print('{} \t {:.4f} \t {:.4f}'.format(ratio, corr, p_value))
                try:
                    plot_local_score_auc_random_plus_active(ratio_df=ratio_df, al=al, df=df, ratio=ratio,
                                                            plot_save_dir=plot_save_dir, dataset=dataset,
                                                            # show=True,
                                                            )
                    plot_local_score_auc_random(ratio_df=ratio_df, al=al, df=df, ratio=ratio,
                                                plot_save_dir=plot_save_dir, dataset=dataset,
                                                figsize=(12, 12),
                                                )
                except:
                    warnings.warn('{} not supported ratio!'.format(str(ratio)))

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

            # Plot class distribution
        plot_distribution_histogram = True
        if plot_distribution_histogram:
            uniform_label = list(range(10)) * 100
            plot_class_dist(uniform_label, ylim=(None, 200))
            # ratio = 0.01  # small ratio
            # ratio = 0.20  # small ratio for vaal only
            ratio = 0.5  # large ratio
            random_label = df[(df['acquisition'] == 'Random') & (df['ratio'] == ratio) & (df['trial'] == 1)]['labels'][
                0]
            plot_class_dist(random_label, ratio=ratio)
            active_indices = df[(df['acquisition'] == al) &
                                (df['ratio'] == ratio) &
                                (df['trial'] == 1)]['indices']
            label_list = []
            for indices_list in active_indices:
                label_list.append([test_data[index][-1] for index in indices_list])
            active_label = label_list
            plot_class_dist(active_label, acquisition=al, ratio=ratio, figsize=[25.4, 57.2])

        # Correlation analysis
        ratios = np.sort(df['ratio'].unique())
        corr_list, p_value_list = [], []
        print('{} \t {} \t {}'.format('ratio', 'corr', 'p_value'))
