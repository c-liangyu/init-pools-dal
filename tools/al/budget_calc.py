import numpy as np


def calc_init_budget(x):
    img_per_class = int(1.052*x+41.33)
    init_budget = int(x * img_per_class)
    return img_per_class, init_budget


if __name__ == '__main__':
    x = 7
    img_per_class, init_budget = calc_init_budget(x)
    print('num_classes: {}'.format(x))
    print('img_per_class: {}, init_budget: {}'.format(img_per_class, init_budget))

