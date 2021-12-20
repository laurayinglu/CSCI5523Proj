#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch

# FILES INFO
DATA_DIR = "/glade/scratch/leikuny/UAV_cashew/DATA_0927" #c
RAW_DATA_DIR = os.path.join(DATA_DIR, "RAW_DATA")
NUMPY_DIR = os.path.join(DATA_DIR, "NUMPY")
RESULT_DIR = os.path.join(DATA_DIR, "RESULT")
MODEL_DIR = os.path.join(DATA_DIR, "MODEL")

# TILES INFO
filename = "/glade/work/leikuny/UAV_cashew/0927/trainData_1.csv" #$
df = pd.read_csv(filename, header=None)
tiles = [file[0].split(".")[0] for file in df.values]

# TRAIN VALIDATION TEST INFO
grids = 5
train_percent = 0.6
validation_percent = 0.2

# LABELS INFO
class_names = ["SAVANNA", "WELL-MANAGED CASHEW", "URBAN", "DENSE FOREST", "BACKGROUND"]
classes = len(class_names)
unknown_class = 10
iterations = 2  # erosion
threshold = 30  # connected_components
prob_upper_threshold = 0.9
prob_lower_threshold = 0.4

# CHANNELS INFO
clip = 1
channel_names = ["BLUE", "GREEN", "RED"]
channels = len(channel_names)

# TIME SERIES INFO
steps = ["202001"]
time_steps = len(steps)
step = 0

# INPUT OUTPUT INFO
input_patch_size = 256
output_patch_size = 240
stride = input_patch_size//2
diff = (input_patch_size-output_patch_size)//2
batch_size = 100
block_size = 2000

# MODEL INFO
device = torch.device("cuda")
model_name = "UNET"

n_epochs = 200
if model_name == "UNET":
    n_epochs = 100
    learning_rate = 0.0001

# SCORING FUNCTIONS
def iou(prediction,label,c):
    union = 0
    inter = 0
    h = prediction.shape[0]
    for i in range(h):
        if (prediction[i] == c and label[i] == c):
            inter = inter + 1
        if (prediction[i] == c or label[i] == c):
            union = union + 1
    iou = inter/union
    return iou

#T DATA_LOADER
TRAIN = True
