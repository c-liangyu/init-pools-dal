from datetime import datetime
import numpy as np
import pandas as pd
import scipy.stats
from scipy import spatial

import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import itertools
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import pycls.core.builders as model_builder
import pycls.utils.checkpoint as cu
from pycls.core.config import cfg
from pycls.datasets.data import Data
import plotly.graph_objects as go
import plotly.io as pio
from pandarallel import pandarallel
from torchvision import transforms

import glob

pandarallel.initialize()

pio.renderers.default = "png"


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning Cold Start Benchmark')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    return parser


def read_dir(path):
    file_list = glob.glob(path+'*.out')
    file_info = {}
    for file in file_list:
        file_info[file] = {'trial': file.split('-')[1], 'ratio': file.split('-p')[-1].split('.out')[0]}

        f = open(file, "r")
        lines = f.readlines()

        for idx, line in enumerate(lines):
            if line.endswith('test samples\n'):
                file_info[file]['indices'] = [int(index) for index in list(lines[idx+1][1:-2].split(','))]
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
    if al == 'entropy':
        score = entropy(cfg, dataObj=dataObj, model=model, dataset=dataset)
    elif al == 'coreset':
        pass
    elif al == 'margin':
        score = margin(cfg, dataObj=dataObj, model=model, dataset=dataset)
    elif al == 'bald':
        score = bald(cfg, dataObj=dataObj, model=model, dataset=dataset)
    elif al == 'consistency':
        score = consistency(cfg, dataObj=dataObj, model=model, dataset=dataset)
    elif al == 'vaal':
        score = vaal(cfg, dataObj=dataObj, model=model, dataset=dataset)
    return score


def entropy(cfg, dataObj, model, dataset):

    """
    Implements the uncertainty principle as a acquisition function.
    """
    if gpu is True:
        num_classes = cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model.cuda()

        u_ranks = []
        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     print("Loading the model in data parallel where num_GPUS: {}".format(self.cfg.NUM_GPUS))
        #     clf = torch.nn.DataParallel(clf, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
        #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        uSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(dataset)), batch_size=int(cfg.TRAIN.BATCH_SIZE), data=dataset)

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)

                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank = temp_u_rank * torch.log2(temp_u_rank)
                temp_u_rank = -1*torch.sum(temp_u_rank, dim=1)
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
    sorted_idx = np.argsort(u_ranks)[::-1]  # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
    np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'uncertainty_sorted_idx.npy'), sorted_idx)
    return u_ranks


def consistency(cfg, dataObj, model, dataset):
    if gpu is True:
        # config dataloaders
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

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
            uSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(dataset)), batch_size=int(cfg.TRAIN.BATCH_SIZE),
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
    sorted_idx = np.argsort(var)[::-1]  # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
    np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'consistency_sorted_idx.npy'), sorted_idx)
    return var


def margin(cfg, dataObj, model, dataset):

    """
    Implements the uncertainty principle as a acquisition function.
    """
    if gpu is True:
        num_classes = cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model.cuda()

        u_ranks = []
        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     print("Loading the model in data parallel where num_GPUS: {}".format(self.cfg.NUM_GPUS))
        #     clf = torch.nn.DataParallel(clf, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
        #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        uSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(dataset)), batch_size=int(cfg.TRAIN.BATCH_SIZE), data=dataset)

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
    sorted_idx = np.argsort(u_ranks)[::-1] # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
    np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'magrin_sorted_idx.npy'), sorted_idx)
    return u_ranks


def vaal(cfg, dataObj, model, dataset):
    if gpu is True:
        clf = model.cuda()
        lSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(train_data)), batch_size=int(cfg.TRAIN.BATCH_SIZE),
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
    sorted_idx = np.argsort(score)[::-1]  # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
    np.save(os.path.join(df_save_dir, dataset.__class__.__name__, 'vaal_sorted_idx.npy'), sorted_idx)
    return score


def get_predictions(clf_model, dataObj, idx_set, dataset):

    clf_model.cuda()
    #Used by bald acquisition
    # if self.cfg.TRAIN.DATASET == "IMAGENET":
    #     tempIdxSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=idx_set, isDistributed=False, isShuffle=False, isVaalSampling=False)
    # else:
    tempIdxSetLoader = dataObj.getSequentialDataLoader(indexes=np.array(idx_set), batch_size=int(cfg.TRAIN.BATCH_SIZE/cfg.NUM_GPUS),data=dataset)

    preds = []
    for i, (x, _) in enumerate(tqdm(tempIdxSetLoader, desc="Collecting predictions in get_predictions function")):
        with torch.no_grad():
            x = x.cuda()
            x = x.type(torch.cuda.FloatTensor)

            temp_pred = clf_model(x)

            #To get probabilities
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
            dropout_score = get_predictions(clf_model=clf_model, dataObj=dataObj, idx_set=np.arange(len(dataset)), dataset=dataset)

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
    return U_X


def coreset(cfg, dataObj, model, dataset):
    if gpu is True:
        clf = model.cuda()

        uSetLoader = dataObj.getSequentialDataLoader(indexes=np.arange(len(dataset)), batch_size=int(cfg.TRAIN.BATCH_SIZE),
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
        # checkpoint_file = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/CIFAR10_REVERSE/resnet18/reverse_trial1/episode_0/vlBest_acc_78_model_epoch_0180.pyth'
        checkpoint_file = os.path.join(df_save_dir, 'vlBest_acc_78_model_epoch_0180.pyth')
        mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]
    elif dataset == 'PATHMNIST_REVERSE':
        DATA_DIR = '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/pathmnist_random_selection_wt_imagenet/logs/'
        # checkpoint_file = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/PATHMNIST_REVERSE/resnet18/trial1/episode_0/vlBest_acc_94_model_epoch_0200.pyth'
        checkpoint_file = os.path.join(df_save_dir, 'vlBest_acc_94_model_epoch_0200.pyth')
        mean, std = [.5, .5, .5], [.5, .5, .5]

    gpu = True
    # gpu = False

    # Active learning selection strategy
    # al = 'entropy'
    # al = 'coreset'
    # al = 'margin'
    # al = 'bald'
    al = 'consistency'
    # al = 'vaal'

    # Create df
    if os.path.isfile(os.path.join(df_save_dir, dataset+'.pkl')):
        df = pd.read_pickle(os.path.join(df_save_dir, dataset+'.pkl'))
    else:
        file_info = read_dir(DATA_DIR)
        df = pd.DataFrame.from_dict(file_info, orient='index')
        df = df.astype({'auc': 'float64', 'ratio': 'float64', 'trial': 'float64'})
        df = df.dropna()
        df.to_pickle(os.path.join(df_save_dir, dataset+'.pkl'))
    print(df.dtypes)

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

    if al is 'coreset':
        features = coreset(cfg, data_obj, model, test_data)
        df = df[df['ratio'] <= 0.005]
        df['local_score'] = df['indices'].parallel_apply(calculate_coreset_score)/len(df['indices'])  # reduce score by average
    else:
        score = calculate_global_score(al, cfg, data_obj, model, test_data)

        sns.displot(x=score)
        plt.savefig(os.path.join(plot_save_dir, dataset, al+'_score.png'))

        df['local_score'] = df['indices'].apply(calculate_local_score)/len(df['indices'])  # reduce score by average

    # Correlation analysis
    ratios = np.sort(df['ratio'].unique())
    corr_list, p_value_list = [], []
    print('{} \t {} \t {}'.format('ratio', 'corr', 'p_value'))
    for ratio in ratios:
        if al is 'coreset' and ratio > 0.005:
            break
        ratio_df = df[df['ratio'] == ratio]
        corr, p_value = scipy.stats.pearsonr(ratio_df['auc'], ratio_df['local_score'])

        sns.displot(x=ratio_df['local_score'])
        plt.title(al+str(ratio))
        Path(os.path.join(plot_save_dir, dataset, al)).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(plot_save_dir, dataset, al, str(ratio)+'_histogram.png'))
        plt.show()

        corr_list.append(corr)
        p_value_list.append(p_value)
        print('{} \t {:.4f} \t {:.4f}'.format(ratio, corr, p_value))

        sns.scatterplot(x=ratio_df['auc'], y=ratio_df['local_score'])
        plt.title(al+str(ratio))
        Path(os.path.join(plot_save_dir, dataset, al)).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(plot_save_dir, dataset, al, str(ratio)+'.png'))
        plt.show()

    sns.lineplot(x=ratios, y=corr_list)
    sns.lineplot(x=ratios, y=p_value_list)
    plt.title(al)
    plt.show()
    plt.savefig(os.path.join(plot_save_dir, dataset, 'summary.png'))
    print()