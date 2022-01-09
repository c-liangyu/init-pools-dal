import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import  torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image, make_grid
from PIL import Image
import matplotlib.pyplot as plt



ratio = 0.2
col = 52

file = 'plot_episode_yvalues.txt'
initial_pool_file = 'lSet.npy'
trial = 'al_cifar10_random_'+str(ratio)+'percent_start_only__deterministic_trial'+str(col+1)
# trial = 'al_cifar10_random_'+str(ratio)+'percent_im_start_only__deterministic_trial'+str(col)
dir = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/CIFAR10/resnet18/'
# dir = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/output/IMBALANCED_CIFAR10/resnet18/'

path = os.path.join(dir, trial, file)
with open(path) as f:
    lines = f.readlines()
    result = float(lines[0])
lSet = np.load(os.path.join(dir, trial, initial_pool_file), allow_pickle=True).astype(int)

print(lSet)

root = '/media/ntu/volume2/home/s121md302_06/workspace/code/init-pools-dal/data/cifar-10'
train = True
dataset = datasets.CIFAR10(root, train=train, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
labels = []
# with torch.no_grad():
#     val_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)
#     pbar = tqdm(total=len(val_loader))
#     for batch in val_loader:
#         image, label = batch[1], batch[-1]
#         # image = image.to(device)
#         # mean, score = system.forward(image)
#         # norms.append(score.squeeze(-1).cpu().numpy())
#         labels.append(label.cpu().numpy())
#         pbar.update()
#     pbar.close()
images, labels = list(zip(*[dataset.__getitem__(index) for index in lSet]))
images, labels = list(images), list(labels)
lSet_labels = np.hstack(labels).astype(int)
lSet_labels_count = np.bincount(lSet_labels)
lSet_labels_count_dict = dict(zip(dataset.class_to_idx.keys(), lSet_labels_count))
print(lSet_labels_count_dict)
lSet_labels_count_norm = lSet_labels_count/lSet_labels_count.sum()
lSet_labels_count_entropy = -np.sum(lSet_labels_count_norm * np.log2(lSet_labels_count_norm))
print('Entropy of class distribution: {:.2f}'.format(lSet_labels_count_entropy))


N = 6
all_low = []
for label in [0, 3, 4, 5, 7, 8]:
    indices_l = np.where(lSet_labels == label)[0]
    indices_l = lSet[indices_l]
    low = indices_l[:N]
    low_images = []
    for index in low:
        image, label = dataset.__getitem__(index)
        # plt.imshow(image.permute(1, 2, 0))
        # plt.show()
        low_images.append(image)
    all_low.extend(low_images)

all_low = make_grid(all_low, nrow=N)
save_image(all_low, os.path.join(dir, trial, 'images_random_sampling.png'))
print(os.path.join(dir, trial, 'images_random_sampling.png'))