#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import config
import os
import numpy as np
import torch
from CONVLSTM import ConvLSTM

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class LSTM(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LSTM,self).__init__()

        self.lstm = torch.nn.LSTM(in_channels, 64, batch_first=True)
        self.linear = torch.nn.Linear(64, out_channels)

        self.relu = torch.nn.ReLU()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self,x):
        x = x[:,:,:,config.diff:-config.diff,config.diff:-config.diff]
        x = torch.reshape(x, (-1, config.time_steps, config.channels, x.shape[3]*x.shape[4]))
        x = x.permute(0,3,1,2)
        x = x.reshape(x.shape[0]*x.shape[1], config.time_steps, config.channels)

        x_lstm, _ = self.lstm(x)
        x_lstm = self.relu(x_lstm)
        out = self.linear(torch.mean(x_lstm,dim = 1))
        out = out.view(-1,config.output_patch_size, config.output_patch_size, config.classes)
        out = out.permute(0,3,1,2)
        return out


class CALD(LSTM):
    def __init__(self, in_channels, out_channels):
        super(CALD,self).__init__(in_channels, out_channels)

        self.attention = torch.nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self,x):
        x = x[:,:,:,config.diff:-config.diff,config.diff:-config.diff]
        x = torch.reshape(x, (-1, config.time_steps, config.channels, x.shape[3]*x.shape[4]))
        x = x.permute(0,3,1,2)
        x = x.reshape(x.shape[0]*x.shape[1], config.time_steps, config.channels)

        x_lstm, _ = self.lstm(x)
        x_lstm = self.relu(x_lstm)
        x_lstm = torch.reshape(x_lstm, (-1,64))
        temporal_att = torch.nn.functional.softmax(self.attention(x_lstm).view(-1, config.time_steps), dim=1)
        temporal_context = torch.sum((temporal_att.view(-1, 1)*x_lstm).view(-1, config.time_steps, 64), dim=1)
        out = self.linear(temporal_context)
        out = out.view(-1,config.output_patch_size, config.output_patch_size, config.classes)
        out = out.permute(0,3,1,2)
        return out


class UNET(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNET,self).__init__()
        self.conv1_1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.conv4_1 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv4_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.conv5_1 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.conv5_2 = torch.nn.Conv2d(256, 256, 3, padding=1)

        self.unpool4 = torch.nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2)
        self.upconv4_1 = torch.nn.Conv2d(256, 128, 3, padding=1)
        self.upconv4_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.unpool3 = torch.nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2)
        self.upconv3_1 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.upconv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.unpool2 = torch.nn.ConvTranspose2d(64 , 32, kernel_size=2, stride=2)
        self.upconv2_1 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.upconv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.unpool1 = torch.nn.ConvTranspose2d(32 , 16, kernel_size=2, stride=2)
        self.upconv1_1 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.upconv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)

        self.out = torch.nn.Conv2d(16, out_channels, kernel_size=1, padding=0)

        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def crop_and_concat(self, x1, x2):
        x1_shape = x1.shape
        x2_shape = x2.shape
        offset_2, offset_3 = (x1_shape[2]-x2_shape[2])//2, (x1_shape[3]-x2_shape[3])//2
        x1_crop = x1[:, :, offset_2:offset_2+x2_shape[2], offset_3:offset_3+x2_shape[3]]
        return torch.cat([x1_crop, x2], dim=1)

    def forward(self, x, time_step):
        x = x[:,time_step,:,:,:]

        conv1 = self.relu(self.conv1_2(self.relu(self.conv1_1(x))))
        maxpool1 = self.maxpool(conv1)
        conv2 = self.relu(self.conv2_2(self.relu(self.conv2_1(maxpool1))))
        maxpool2 = self.maxpool(conv2)
        conv3 = self.relu(self.conv3_2(self.relu(self.conv3_1(maxpool2))))
        maxpool3 = self.maxpool(conv3)
        conv4 = self.relu(self.conv4_2(self.relu(self.conv4_1(maxpool3))))
        maxpool4 = self.maxpool(conv4)
        conv5 = self.relu(self.conv5_2(self.relu(self.conv5_1(maxpool4))))

        unpool4 = self.unpool4(conv5)
        upconv4 = self.relu(self.upconv4_2(self.relu(self.upconv4_1(self.crop_and_concat(conv4, unpool4)))))
        unpool3 = self.unpool3(upconv4)
        upconv3 = self.relu(self.upconv3_2(self.relu(self.upconv3_1(self.crop_and_concat(conv3, unpool3)))))
        unpool2 = self.unpool2(upconv3)
        upconv2 = self.relu(self.upconv2_2(self.relu(self.upconv2_1(self.crop_and_concat(conv2, unpool2)))))
        unpool1 = self.unpool1(upconv2)
        upconv1 = self.relu(self.upconv1_2(self.relu(self.upconv1_1(self.crop_and_concat(conv1, unpool1)))))

        out = self.out(upconv1)
        return out[:,:,config.diff:-config.diff, config.diff:-config.diff]

if __name__ == "__main__":
    data = torch.randn(config.batch_size, config.time_steps, config.channels, config.input_patch_size, config.input_patch_size)
    print(data.shape, "DATA")

    model = LSTM(in_channels=config.channels, out_channels=config.classes)
    model = model.to('cuda')
    out = model(data.to('cuda'))
    print(data.shape, out.shape, "LSTM")

    model = CALD(in_channels=config.channels, out_channels=config.classes)
    model = model.to('cuda')
    out = model(data.to('cuda'))
    print(data.shape, out.shape, "CALD")

    model = UNET(in_channels=config.channels, out_channels=config.classes)
    model = model.to('cuda')
    out = model(data.to('cuda'), 1)
    print(data.shape, out.shape, "UNET")