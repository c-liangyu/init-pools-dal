import numpy as np 
import torch
from statistics import mean
import gc
import os
import math
import sys
import time
import pickle
import math
from copy import deepcopy
from tqdm import tqdm

from scipy.spatial import distance_matrix
import torch.nn as nn

class EntropyLoss(nn.Module):
    """
    This class contains the entropy function implemented.
    """
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x, applySoftMax=True):
        #Assuming x : [BatchSize, ]

        if applySoftMax:
            entropy = torch.nn.functional.softmax(x, dim=1)*torch.nn.functional.log_softmax(x, dim=1)
        else:
            entropy = x * torch.log2(x)
        entropy = -1*entropy.sum(dim=1)
        return entropy 

class SelfSupervisionSampling:
    """
    Loads SimCLR and VAE losses. Generates the initial pool accordingly. 
    """

    def __init__(self, cfg, dataset, budgetSize, sampling_fn, dataset_name):
        fullSet = np.array([i for i in range(len(dataset))],dtype=np.ndarray)

        output_dir = '../results/'
        if dataset_name == 'CIFAR10':
            output_dir += 'cifar-10/'
            num_clusters = 10
        elif dataset_name == 'CIFAR100':
            output_dir += 'cifar-100/'
            num_clusters = 19
        elif dataset_name == 'MNIST':
            output_dir += 'mnist/'
            num_clusters = 10
        elif dataset_name == 'TINYIMAGENET':
            output_dir += 'tinyimagenet/'
            num_clusters = 200
        elif dataset_name == 'IMBALANCED_CIFAR10':
            output_dir += 'imbalanced-cifar-10/'
            num_clusters = 10

        if sampling_fn == "simclr":
            # file_path = f'{output_dir}/{dataset_name}_SimCLR_losses_{cfg.INIT_POOL.SIMCLR_DUPLICATE}.npy'
            # file_path = f'{output_dir}/{cfg.INIT_POOL.SIMCLR_DUPLICATE}'
            file_path = f'{cfg.INIT_POOL.SIMCLR_DUPLICATE}'
        elif sampling_fn == "vae":
            file_path = f'{output_dir}/{dataset_name}_VAE_losses.npy'

        if sampling_fn == 'determined':
            file_path = f'{cfg.INIT_POOL.SIMCLR_DUPLICATE}'
            initSet = np.load(file_path)[:cfg.INIT_POOL.INIT_NUMBER]
            self.initSet = fullSet[initSet]
            self.remainSet = np.delete(fullSet, initSet)
        # Sample with class balance
        elif cfg.INIT_POOL.CLUSTER_ID is not None:
            cluster_ids = np.load(f'{output_dir}/{cfg.INIT_POOL.CLUSTER_ID}')
            losses = np.load(file_path)
            sorted_idx = np.argsort(losses)[::-1]
            cluster_ids = cluster_ids[sorted_idx]
            # Equal budget assigned to all classes
            cluster_budgets = [int(budgetSize / num_clusters) for x in range(num_clusters)]
            groups = []
            if dataset_name == 'CIFAR100':
                num_clusters += 1
            for cluster_id in range(num_clusters):
                if cluster_id == 1 and dataset_name == 'CIFAR100':
                    continue
                groups.append(np.array([idx for idx, x in enumerate(cluster_ids) if x == cluster_id]))
            self.initSet = []
            self.remainSet = []
            # Filter from full buckets
            for idx, g in enumerate(groups):
                self.initSet.extend(sorted_idx[g[:cluster_budgets[idx]]])
                self.remainSet.extend(sorted_idx[g[cluster_budgets[idx]:]])
        else:
            losses = np.load(file_path)
            sorted_idx = np.argsort(losses)[::-1]
            initSet = sorted_idx[:budgetSize]
            self.initSet = fullSet[initSet]
            self.remainSet = fullSet[sorted_idx[budgetSize:]]


    def sample(self):
        return self.initSet, self.remainSet 


class ClusteringSampling:
    """
    Loads SCAN and K-Means cluster ids. Generates the initial pool accordingly.
    """
    def __init__(self, dataset, budgetSize, sampling_fn, dataset_name):
        fullSet = [i for i in range(len(dataset))]

        output_dir = '../results/'
        if dataset_name == 'CIFAR10':
            output_dir += 'cifar-10'
            num_clusters = 10
        elif dataset_name == 'CIFAR100':
            output_dir += 'cifar-100'
            num_clusters = 19
        elif dataset_name == 'MNIST':
            output_dir += 'mnist'
            num_clusters = 10
        elif dataset_name == 'TINYIMAGENET':
            output_dir += 'tinyimagenet'
            num_clusters = 200
        elif dataset_name == 'IMBALANCED_CIFAR10':
            output_dir += 'imbalanced-cifar-10/'
            num_clusters = 10
        
        if sampling_fn == "scan":
            file_path = f'{output_dir}/{dataset_name}_SCAN_cluster_ids.npy'
        elif sampling_fn == "kmeans":
            file_path = f'{output_dir}/{dataset_name}_kmeans_cluster_ids.npy'
        
        cluster_ids = np.load(file_path)
        # Equal budget assigned to all classes
        cluster_budgets = [int(budgetSize/num_clusters) for x in range(num_clusters)]
        groups = []
        if dataset_name == 'CIFAR100':
            num_clusters += 1
        for cluster_id in range(num_clusters):
            if cluster_id == 1 and dataset_name == 'CIFAR100':
                continue
            groups.append(np.array([idx for idx, x in enumerate(cluster_ids) if x == cluster_id]))
        self.initSet = []
        self.remainSet = []
        for idx, g in enumerate(groups):
            np.random.shuffle(g)
            self.initSet.extend(g[:cluster_budgets[idx]])
            self.remainSet.extend(g[cluster_budgets[idx]:])


    def sample(self):
        return self.initSet, self.remainSet 