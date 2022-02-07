import numpy as np

if __name__ == '__main__':
    # path = '../results/cifar-10/CIFAR10_SimCLR_losses_10.npy'
    # path = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/CIFAR10/resnet18/al_cifar10_random_0.2percent_start_only__deterministic_trial2/lSet.npy'
    # path = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/IMBALANCED_CIFAR10/resnet18/al_cifar10_random_2percent_im_start_only_trial1/lSet.npy'
    # path = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/CIFAR10/resnet18/al_cifar10_random_0.2percent_start_only_trial67/lSet.npy'
    # paths = ['/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/CIFAR10/resnet18/al_cifar10_random_2percent_start_only_trial'+str(trial)+'/lSet.npy' for trial in range(1, 101)]
    # paths = ['/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/CIFAR10/resnet18/al_cifar10_random_0.2percent_start_only__deterministic_trial'+str(trial)+'/lSet.npy' for trial in range(1, 14)]
    # paths = ['/media/ntu/volume2/home/s121md302_06/workspace/code/bilevel_coresets/data_summarization/results/inds_None.npy']
    paths = [
        '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/df/PathMNIST/coreset_sorted_idx.npy',
        '/media/ntu/volume2/home/s121md302_06/workspace/data/cold_bench/df/PathMNIST/consistency_sorted_idx.npy',
             ]
    # path = '/home/students/dipe051-1/workspace/code/init-pools-dal/results/cifar-10/CIFAR10_SimCLR_losses.npy'
    for idx, path in enumerate(paths):
        print(idx)
        file = np.load(path, allow_pickle=True)
        print(file.shape)
        print(file)

