#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import os
import config
import copy
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from collections import Counter
import torch
from torch.utils.data.dataset import Dataset
from osgeo import gdal, gdalconst, osr

if not os.path.exists(config.NUMPY_DIR):
    os.makedirs(config.NUMPY_DIR)
if not os.path.exists(config.RESULT_DIR):
    os.makedirs(config.RESULT_DIR)
if not os.path.exists(config.MODEL_DIR):
    os.makedirs(config.MODEL_DIR)
# %%
def create_patches(grid):
    image = np.load(os.path.join(config.NUMPY_DIR, grid+"_image.npy"))
    image_valid = np.load(os.path.join(config.NUMPY_DIR, grid+"_image_valid.npy"))
    label = np.load(os.path.join(config.NUMPY_DIR, grid+"_label.npy"))
    label_valid = np.load(os.path.join(config.NUMPY_DIR, grid+"_label_valid.npy"))
    height, width = label.shape
    diff = (config.input_patch_size-config.output_patch_size)//2
    image_patches = []
    label_patches = []
    for i in range(height//config.output_patch_size):
        for j in range(width//config.output_patch_size):
            i_label_start, i_label_end = i*config.output_patch_size, (i+1)*config.output_patch_size
            i_image_start, i_image_end = i_label_start-diff, i_label_end+diff
            j_label_start, j_label_end = j*config.output_patch_size, (j+1)*config.output_patch_size
            j_image_start, j_image_end = j_label_start-diff, j_label_end+diff
            if 0<i_image_start<height and 0<i_image_end<height and 0<j_image_start<width and 0<j_image_end<width:
                image_patch = image[:, :, i_image_start:i_image_end, j_image_start:j_image_end]
                label_patch = label[i_label_start:i_label_end, j_label_start:j_label_end]
                image_valid_patch = image_valid[i_image_start:i_image_end, j_image_start:j_image_end]
                label_valid_patch = label_valid[i_label_start:i_label_end, j_label_start:j_label_end]
                if np.sum(image_valid_patch) == config.input_patch_size*config.input_patch_size:
                    image_patches.append(image_patch)
                    label_patches.append(label_patch)
                    stats = Counter(np.reshape(label_patch, (-1)))
    image_patches = np.array(image_patches).astype(np.float32)
    label_patches = np.array(label_patches).astype(np.int8)
    return image_patches, label_patches

if config.TRAIN == True:
    class SEGMENTATION(Dataset):

        def __init__(self, image_patches, label_patches):
            self.image_patches = image_patches
            self.label_patches = label_patches

        def __len__(self):
            return len(self.image_patches)

        def __getitem__(self, index):
            item = self.image_patches[index], self.label_patches[index]
            return item

if config.TRAIN == False:
    class SEGMENTATION(Dataset):

        def __init__(self, image_patches):
            self.image_patches = image_patches

        def __len__(self):
            return len(self.image_patches)

        def __getitem__(self, index):
            return self.image_patches[index]

# %%
if __name__ == "__main__":
    dataset = np.load(os.path.join(config.NUMPY_DIR, "train_set.npy"))
    dataset = dataset[random.sample(range(len(dataset)), len(dataset))]
    for grid in dataset:
        print(grid)
        image_patches, label_patches = create_patches(grid)
        print(image_patches.shape, label_patches.shape)
        break

    for epoch in range(1):
        dataset = np.load(os.path.join(config.NUMPY_DIR, "train_set.npy"))
        dataset = dataset[random.sample(range(len(dataset)), len(dataset))]
        for grid in dataset:
            print(grid)
            image_patches, label_patches = create_patches(grid)
            data = SEGMENTATION(image_patches, label_patches)
            data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.batch_size, shuffle=True, num_workers=0)
            for batch, [image_patch, label_patch] in enumerate(data_loader):
                print(image_patch.shape, label_patch.shape)
            break
