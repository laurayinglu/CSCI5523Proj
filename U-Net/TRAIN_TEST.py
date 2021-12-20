#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import config_0924 as config
import os
import numpy as np
from collections import Counter
import random

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

import MODEL.MODEL as MODEL
import DATA_LOADER
# %%
print("#######################################################################")
print("BUILD MODEL:{}".format(config.model_name))
model_name = config.model_name
model = getattr(MODEL, model_name)(in_channels=config.channels, out_channels=config.classes)
model = model.to('cuda')
model = torch.nn.DataParallel(model, device_ids=[0, 1])
criterion = torch.nn.CrossEntropyLoss(ignore_index=config.unknown_class, reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
# %%
print("#######################################################################")
print("MAKE DIRECTORIES")
if not os.path.exists(os.path.join(config.MODEL_DIR, "TRAIN")):
    os.makedirs(os.path.join(config.MODEL_DIR, "TRAIN"))
if not os.path.exists(os.path.join(config.RESULT_DIR, "TRAIN", model_name)):
    os.makedirs(os.path.join(config.RESULT_DIR, "TRAIN", model_name))
# %%
print("#######################################################################")
print("TRAIN MODEL")
train_loss = []
validation_loss = [10000]
test_loss = []
# train_score = []
# validation_score = []
# test_score = []
# max_score = 0
for epoch in range(1,config.n_epochs+1):

    model.train()
    dataset = np.load(os.path.join(config.NUMPY_DIR, "train_set.npy"))
    dataset = dataset[random.sample(range(len(dataset)), len(dataset))]
    epoch_loss = 0
    for grid_num,grid in enumerate(dataset):
        image_patches, label_patches = DATA_LOADER.create_patches(grid)
        data = DATA_LOADER.SEGMENTATION(image_patches, label_patches)
        data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.batch_size, shuffle=True, num_workers=0)
        grid_loss = 0
        for batch, [image_patch, label_patch] in enumerate(data_loader):
            optimizer.zero_grad()
            out = model(image_patch.to('cuda'), config.step)
            label_patch = label_patch.type(torch.long).to('cuda')

            batch_loss = criterion(out, label_patch)
            batch_loss = torch.masked_select(batch_loss, (label_patch!=config.unknown_class))
            batch_loss = torch.mean(batch_loss)

            batch_loss.backward()
            optimizer.step()
            grid_loss += batch_loss.item()
        grid_loss = grid_loss/(batch+1)
        print('Grid:{}\tGrid Loss:{:.4f}'.format(grid, grid_loss))
        epoch_loss += grid_loss
    epoch_loss = epoch_loss/(grid_num+1)
    print('Epoch:{}\tTrain Loss:{:.4f}'.format(epoch, epoch_loss), end="\t")
    train_loss.append(epoch_loss)

    model.eval()
    with torch.no_grad():
        dataset = np.load(os.path.join(config.NUMPY_DIR, "validation_set.npy"))
        dataset = dataset[random.sample(range(len(dataset)), len(dataset))]
        epoch_loss = 0
        for grid_num,grid in enumerate(dataset):
            image_patches, label_patches = DATA_LOADER.create_patches(grid)
            data = DATA_LOADER.SEGMENTATION(image_patches, label_patches)
            data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.batch_size, shuffle=True, num_workers=0)
            grid_loss = 0
            for batch, [image_patch, label_patch] in enumerate(data_loader):
                out = model(image_patch.to('cuda'), config.step)
                label_patch = label_patch.type(torch.long).to('cuda')

                batch_loss = criterion(out, label_patch)
                batch_loss = torch.masked_select(batch_loss, (label_patch!=config.unknown_class))
                batch_loss = torch.mean(batch_loss)

                grid_loss += batch_loss.item()
            grid_loss = grid_loss/(batch+1)
            epoch_loss += grid_loss
        epoch_loss = epoch_loss/(grid_num+1)
        print('Val Loss:{:.4f}'.format(epoch_loss), end="\t")
        if epoch_loss < min(validation_loss):
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, "TRAIN", model_name + ".pt"))
        validation_loss.append(epoch_loss)

        dataset = np.load(os.path.join(config.NUMPY_DIR, "test_set.npy"))
        dataset = dataset[random.sample(range(len(dataset)), len(dataset))]
        epoch_loss = 0
        for grid_num,grid in enumerate(dataset):
            image_patches, label_patches = DATA_LOADER.create_patches(grid)
            data = DATA_LOADER.SEGMENTATION(image_patches, label_patches)
            data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.batch_size, shuffle=True, num_workers=0)
            grid_loss = 0
            for batch, [image_patch, label_patch] in enumerate(data_loader):
                out = model(image_patch.to('cuda'), config.step)
                label_patch = label_patch.type(torch.long).to('cuda')

                batch_loss = criterion(out, label_patch)
                batch_loss = torch.masked_select(batch_loss, (label_patch!=config.unknown_class))
                batch_loss = torch.mean(batch_loss)

                grid_loss += batch_loss.item()
            grid_loss = grid_loss/(batch+1)
            epoch_loss += grid_loss
        epoch_loss = epoch_loss/(grid_num+1)
        print('Test Loss:{:.4f}'.format(epoch_loss), end="\t")
        test_loss.append(epoch_loss)

    # dataset = np.load(os.path.join(config.NUMPY_DIR, "train_set.npy"))
    # dataset = dataset[random.sample(range(len(dataset)), len(dataset))]
    # total = 0
    # for grid_num,grid in enumerate(dataset):
    #     image_patches, label_patches = DATA_LOADER.create_patches(grid)
    #     data = DATA_LOADER.SEGMENTATION(image_patches, label_patches)
    #     data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    #     if not 'pred_labels' in globals():
    #         pred_labels = np.zeros((len(dataset)*len(data_loader)*config.batch_size, config.output_patch_size, config.output_patch_size)).astype(np.float32)
    #         true_labels = np.zeros((len(dataset)*len(data_loader)*config.batch_size, config.output_patch_size, config.output_patch_size)).astype(np.float32)
    #     for batch, [image_patch, label_patch] in enumerate(data_loader):
    #         out = model(image_patch.to('cuda'))
    #         out = torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1)
    #         pred_labels[total:total+len(image_patch)] = out.detach().cpu().numpy()
    #         true_labels[total:total+len(image_patch)] = label_patch.cpu().numpy()
    #         total += len(image_patch)
    # pred_labels = np.reshape(pred_labels[:total], (-1))
    # true_labels = np.reshape(true_labels[:total], (-1))
    # pred_labels = pred_labels[true_labels!=config.unknown_class]
    # true_labels = true_labels[true_labels!=config.unknown_class]
    # # train_score.append(accuracy_score(y_true=true_labels, y_pred=pred_labels))
    # train_score.append(f1_score(y_true=true_labels, y_pred=pred_labels, average='macro'))
    # print('Train Score:{:.4f}'.format(train_score[-1]), end="\t")
    # del pred_labels
    # del true_labels

    # dataset = np.load(os.path.join(config.NUMPY_DIR, "validation_set.npy"))
    # dataset = dataset[random.sample(range(len(dataset)), len(dataset))]
    # total = 0
    # for grid_num,grid in enumerate(dataset):
    #     image_patches, label_patches = DATA_LOADER.create_patches(grid)
    #     data = DATA_LOADER.SEGMENTATION(image_patches, label_patches)
    #     data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    #     if not 'pred_labels' in globals():
    #         pred_labels = np.zeros((len(dataset)*len(data_loader)*config.batch_size, config.output_patch_size, config.output_patch_size)).astype(np.float32)
    #         true_labels = np.zeros((len(dataset)*len(data_loader)*config.batch_size, config.output_patch_size, config.output_patch_size)).astype(np.float32)
    #     for batch, [image_patch, label_patch] in enumerate(data_loader):
    #         out = model(image_patch.to('cuda'))
    #         out = torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1)
    #         pred_labels[total:total+len(image_patch)] = out.detach().cpu().numpy()
    #         true_labels[total:total+len(image_patch)] = label_patch.cpu().numpy()
    #         total += len(image_patch)
    # pred_labels = np.reshape(pred_labels[:total], (-1))
    # true_labels = np.reshape(true_labels[:total], (-1))
    # pred_labels = pred_labels[true_labels!=config.unknown_class]
    # true_labels = true_labels[true_labels!=config.unknown_class]
    # # validation_score.append(accuracy_score(y_true=true_label_patch, y_pred=pred_label_patch))
    # validation_score.append(f1_score(y_true=true_labels, y_pred=pred_labels, average='macro'))
    # print("Val Score:{:.4f}\tMax Score:{:.4f}".format(validation_score[-1], max_score), end="\t")
    # if max_score<validation_score[-1]:
    #     max_score = validation_score[-1]
    #     torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, "TRAIN", model_name+".pt"))
    # del pred_labels
    # del true_labels

    # dataset = np.load(os.path.join(config.NUMPY_DIR, "test_set.npy"))
    # dataset = dataset[random.sample(range(len(dataset)), len(dataset))]
    # total = 0
    # for grid_num,grid in enumerate(dataset):
    #     image_patches, label_patches = DATA_LOADER.create_patches(grid)
    #     data = DATA_LOADER.SEGMENTATION(image_patches, label_patches)
    #     data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    #     if not 'pred_labels' in globals():
    #         pred_labels = np.zeros((len(dataset)*len(data_loader)*config.batch_size, config.output_patch_size, config.output_patch_size)).astype(np.float32)
    #         true_labels = np.zeros((len(dataset)*len(data_loader)*config.batch_size, config.output_patch_size, config.output_patch_size)).astype(np.float32)
    #     for batch, [image_patch, label_patch] in enumerate(data_loader):
    #         out = model(image_patch.to('cuda'))
    #         out = torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1)
    #         pred_labels[total:total+len(image_patch)] = out.detach().cpu().numpy()
    #         true_labels[total:total+len(image_patch)] = label_patch.cpu().numpy()
    #         total += len(image_patch)
    # pred_labels = np.reshape(pred_labels[:total], (-1))
    # true_labels = np.reshape(true_labels[:total], (-1))
    # pred_labels = pred_labels[true_labels!=config.unknown_class]
    # true_labels = true_labels[true_labels!=config.unknown_class]
    # test_score.append(f1_score(y_true=true_labels, y_pred=pred_labels, average='macro'))
    # print('Test Score:{:.4f}'.format(test_score[-1]))
    # del pred_labels
    # del true_labels

plt.figure(figsize=(10,10))
plt.xlabel("#Epoch", fontsize=50)
plt.plot(train_loss, linewidth=4, label="TRAIN LOSS")
plt.plot(validation_loss, linewidth=4, label="VAL LOSS")
plt.plot(test_loss, linewidth=4, label="TEST LOSS")
plt.legend(loc="upper right", fontsize=40, frameon=False)
plt.tight_layout(pad=0.0,h_pad=0.0,w_pad=0.0)
plt.savefig(os.path.join(config.RESULT_DIR, "TRAIN", model_name, "LOSS.pdf"), format = "pdf")
plt.close()

# plt.figure(figsize=(10,10))
# plt.xlabel("#Epoch", fontsize=50)
# plt.plot(train_score, linewidth=4, label="TRAIN SCORE")
# plt.plot(validation_score, linewidth=4, label="VAL SCORE")
# plt.plot(test_score, linewidth=4, label="TEST SCORE")
# plt.legend(loc="lower right", fontsize=40, frameon=False)
# plt.tight_layout(pad=0.0,h_pad=0.0,w_pad=0.0)
# plt.savefig(os.path.join(config.RESULT_DIR, "TRAIN", model_name, "SCORE.pdf"), format = "pdf")
# plt.close()
