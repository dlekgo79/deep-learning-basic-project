import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
import os
from PIL import Image
import pandas as pd
import random

import time
from tqdm.auto import tqdm

for i in tqdm(range(1000)):
    time.sleep(0.01)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_dataframe = pd.read_csv('C:/deep_basic/2023-final/data.csv')
test_df = pd.read_csv('C:/deep_basic/2023-final/testdata.csv')

from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(train_dataframe, shuffle=True, test_size=0.15, stratify=train_dataframe['Label'])



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, train='train', transform=None):
        if train == 'train':
            self.image_list = []
            self.label_list = []
            self.other_list = []
            path = 'C:/deep_basic/2023-final/dataset/{}/{}'  #
            for index, row in dataframe.iterrows():
                image_path = row['Image']
                image_label = row['Label']
                image_age = row['Age']
                image_gender = row['Gender']
                image_race = row['Race']
                image = Image.open(path.format(image_label, image_path)).convert('RGB')
                # if there is transform, apply transform
                if transform != None:
                    image = transform(image)
                self.image_list.append(image)
                self.label_list.append(image_label)
                self.other_list.append((image_age, image_gender, image_race))

        elif train == 'test':
            self.image_list = []
            self.label_list = []  # 이미지의 경로
            self.other_list = []
            path = 'C:/deep_basic/2023-final/testset/{}'
            for index, row in dataframe.iterrows():
                image_path = row['Image']
                image_gender = row['Gender']
                image_race = row['Race']
                image = Image.open(path.format(image_path)).convert('RGB')
                if transform != None:
                    image = transform(image)
                self.image_list.append(image)
                self.label_list.append(image_path)
                self.other_list.append((image_gender, image_race))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return self.image_list[idx], self.label_list[idx], self.other_list[idx]

from autoaugment import *

train_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(),


    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.602927 ,0.46158618 ,0.39498535], [0.21958497, 0.19605462 ,0.18665607])
    # 이미지의 평균과 표준 편차로 나누어 정규화 하는 것
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.602927 ,0.46158618, 0.39498535],[0.21958497 ,0.19605462, 0.18665607])
])

train_dataset = CustomDataset(train_df, train='train', transform=train_transform)
valid_dataset = CustomDataset(valid_df, train='train', transform=test_transform)
test_dataset = CustomDataset(test_df, train='test', transform=test_transform)


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # convolutional layer (sees 3x224x224 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 16x112x112 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 32x56x56 tensor)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 32x28x28 tensor)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d(2, 2)  # 답작성)

        # 64x 14x14
        self.conv5 = nn.Conv2d(64,64, 3, padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(True)


        #64x14x14
        self.conv6 = nn.Conv2d(64,64, 3, padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU(True)
        self.pool6 = nn.MaxPool2d(2, 2)
        # max pooling layer
        #64x7x7
        self.conv9 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.bn9 = nn.BatchNorm2d(128)
        self.relu9 = nn.ReLU(True)



        # linear layer (64 * 4 * 4 -> 500)
        # 64x7x7
        self.dropout7 = nn.Dropout(0.7)
        self.fc1 = nn.Linear(6272+2, 500)  # 답작성)
        self.bcfc1 = nn.BatchNorm1d(500)
        self.relu7 = nn.ReLU(True)
        # linear layer (500 -> 10)
        self.dropout8 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(500, 5)  # 답작성)
        # dropout layer (p=0.25)


        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.kaiming_uniform_(self.conv1.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv2.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv3.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv4.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv5.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv6.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv9.weight.data)
        torch.nn.init.kaiming_uniform_(self.fc1.weight.data)
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data)


    def forward(self, x, data1, data2):
        # add sequence of convolutional and max pooling layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)



        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.pool6(x)


        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)



        #

        x = x.view(-1, 6272)

        data1 = data1.reshape(-1, 1)
        data2 = data1.reshape(-1, 1)

        x = torch.cat((x, data1), dim=1)
        x = torch.cat((x, data2), dim=1)

        # #
        # x = x.reshape(-1,3136)  # 답작성
        # print(x.shape)
        # print(data1.shape)
        # print(data2.shape)
        # # add dropout layer
        x = self.dropout7(x)  # 답작성
        # add 1st hidden layer, with relu activation function
        x = self.relu7(self.bcfc1(self.fc1(x)))  # 답작성
        # add dropout layer
        x = self.dropout8(x)  # 답작성
        x = self.fc2(x)
        # add 2nd hidden layer, with relu activation function



        return x

class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        # convolutional layer (sees 3x224x224 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 16x112x112 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 32x56x56 tensor)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 32x28x28 tensor)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d(2, 2)  # 답작성)

        # 64x 14x14
        self.conv5 = nn.Conv2d(64,64, 3, padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(True)
        self.pool5 = nn.MaxPool2d(2, 2)  # 답작성)



        # 64x7x7
        self.dropout7 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(3136+2, 500)  # 답작성)
        self.bcfc1 = nn.BatchNorm1d(500)
        self.relu7 = nn.ReLU(True)
        # linear layer (500 -> 10)
        self.dropout8 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(500, 5)  # 답작성)




        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.kaiming_uniform_(self.conv1.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv2.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv3.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv4.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv5.weight.data)

        torch.nn.init.kaiming_uniform_(self.fc1.weight.data)
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data)


    def forward(self, x, data1, data2):
        # add sequence of convolutional and max pooling layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)


        x = x.view(-1,3136)

        data1 = data1.reshape(-1, 1)
        data2 = data1.reshape(-1, 1)

        x = torch.cat((x, data1), dim=1)
        x = torch.cat((x, data2), dim=1)


        x = self.dropout7(x)  # 답작성
        x = self.relu7(self.bcfc1(self.fc1(x)))
        x = self.dropout8(x)  # 답작성
        x = self.fc2(x)



        return x


class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        # convolutional layer (sees 3x224x224 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 16x112x112 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 32x56x56 tensor)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 32x28x28 tensor)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d(2, 2)  # 답작성)

        # 64x 14x14
        self.conv5 = nn.Conv2d(64,64, 3, padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(True)


        #64x14x14
        self.conv6 = nn.Conv2d(64,64, 3, padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU(True)
        self.pool6 = nn.MaxPool2d(2, 2)
        # max pooling layer
        #64x7x7
        # self.conv9 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        # self.bn9 = nn.BatchNorm2d(128)
        # self.relu9 = nn.ReLU(True)



        # linear layer (64 * 4 * 4 -> 500)
        # # 64x7x7
        # self.dropout7 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(3136+2, 1500)  # 답작성)
        # self.bcfc1 = nn.BatchNorm1d(1500)
        # self.relu7 = nn.ReLU(True)
        # # linear layer (500 -> 10)
        # self.dropout8 = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(1500, 500)
        # self.bcfc2 = nn.BatchNorm1d(500)
        # self.relu8 = nn.ReLU(True)
        #
        # self.dropout9 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(500, 5)
        # # dropout layer (p=0.25)


        # linear layer (64 * 4 * 4 -> 500)
        # 64x7x7
        self.dropout7 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(3136+2, 500)  # 답작성)
        self.bcfc1 = nn.BatchNorm1d(500)
        self.relu7 = nn.ReLU(True)
        # linear layer (500 -> 10)
        self.dropout8 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(500, 5)  # 답작성)
        # dropout layer (p=0.25)



        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.kaiming_uniform_(self.conv1.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv2.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv3.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv4.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv5.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv6.weight.data)
        # torch.nn.init.kaiming_uniform_(self.conv9.weight.data)
        torch.nn.init.kaiming_uniform_(self.fc1.weight.data)
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data)


    def forward(self, x, data1, data2):
        # add sequence of convolutional and max pooling layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.pool6(x)

        #
        # x = self.conv9(x)
        # x = self.bn9(x)
        # x = self.relu9(x)

        #
        # x = x.view(-1, 3136)
        x = x.view(-1, 3136)

        data1 = data1.reshape(-1, 1)
        data2 = data1.reshape(-1, 1)

        x = torch.cat((x, data1), dim=1)
        x = torch.cat((x, data2), dim=1)


        x = self.dropout7(x)  # 답작성
        x = self.relu7(self.bcfc1(self.fc1(x)))
        x = self.dropout8(x)  # 답작성
        # x = self.relu8(self.bcfc2(self.fc2(x)))
        # x = self.dropout9(x)  # 답작성
        x = self.fc2(x)



        return x



class ConvNet4(nn.Module):
    def __init__(self):
        super(ConvNet4, self).__init__()
        # convolutional layer (sees 3x224x224 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 16x112x112 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 32x56x56 tensor)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 32x28x28 tensor)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d(2, 2)  # 답작성)

        # 64x 14x14
        self.conv5 = nn.Conv2d(64,64, 3, padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(True)
        self.pool5 = nn.MaxPool2d(2, 2)  # 답작성)



        # 64x7x7
        self.dropout7 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(3136+2, 500)  # 답작성)
        self.bcfc1 = nn.BatchNorm1d(500)
        self.relu7 = nn.ReLU(True)
        # linear layer (500 -> 10)
        self.dropout8 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(500, 5)  # 답작성)


        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.kaiming_uniform_(self.conv1.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv2.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv3.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv4.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv5.weight.data)

        torch.nn.init.kaiming_uniform_(self.fc1.weight.data)
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data)


    def forward(self, x, data1, data2):
        # add sequence of convolutional and max pooling layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)


        x = x.view(-1,3136)

        data1 = data1.reshape(-1, 1)
        data2 = data1.reshape(-1, 1)

        x = torch.cat((x, data1), dim=1)
        x = torch.cat((x, data2), dim=1)


        x = self.dropout7(x)  # 답작성
        x = self.relu7(self.bcfc1(self.fc1(x)))
        x = self.dropout8(x)  # 답작성
        x = self.fc2(x)



        return x

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # convolutional layer (sees 3x224x224 image tensor)
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=(1, 1))
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu1 = nn.ReLU(True)
#         self.pool1 = nn.MaxPool2d(2, 2)
#
#         # convolutional layer (sees 16x112x112 tensor)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=(1, 1))
#         self.bn2 = nn.BatchNorm2d(32)
#         self.relu2 = nn.ReLU(True)
#         self.pool2 = nn.MaxPool2d(2, 2)
#
#         # convolutional layer (sees 32x56x56 tensor)
#         self.conv3 = nn.Conv2d(32, 32, 3, padding=(1, 1))
#         self.bn3 = nn.BatchNorm2d(32)
#         self.relu3 = nn.ReLU(True)
#         self.pool3 = nn.MaxPool2d(2, 2)
#
#         # convolutional layer (sees 32x28x28 tensor)
#         self.conv4 = nn.Conv2d(32, 64, 3, padding=(1, 1))
#         self.bn4 = nn.BatchNorm2d(64)
#         self.relu4 = nn.ReLU(True)
#         self.pool4 = nn.MaxPool2d(2, 2)  # 답작성)
#
#         # 64x 14x14
#         self.conv5 = nn.Conv2d(64,64, 3, padding=(1, 1))
#         self.bn5 = nn.BatchNorm2d(64)
#         self.relu5 = nn.ReLU(True)
#         self.pool5 = nn.MaxPool2d(2, 2)  # 답작성)
#
#         # 64x7x7
#         self.conv6 = nn.Conv2d(64,128, 3, padding=(1, 1))
#         self.bn6 = nn.BatchNorm2d(128)
#         self.relu6 = nn.ReLU(True)
#
#         # 128x7x7
#         self.dropout7 = nn.Dropout(0.25)
#         self.fc1 = nn.Linear(6272+2, 500)  # 답작성)
#         self.bcfc1 = nn.BatchNorm1d(500)
#         self.relu7 = nn.ReLU(True)
#         # linear layer (500 -> 10)
#         self.dropout8 = nn.Dropout(0.25)
#         self.fc2 = nn.Linear(500, 5)  # 답작성)
#
#
#
#
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         torch.nn.init.kaiming_uniform_(self.conv1.weight.data)
#         torch.nn.init.kaiming_uniform_(self.conv2.weight.data)
#         torch.nn.init.kaiming_uniform_(self.conv3.weight.data)
#         torch.nn.init.kaiming_uniform_(self.conv4.weight.data)
#         torch.nn.init.kaiming_uniform_(self.conv5.weight.data)
#         torch.nn.init.kaiming_uniform_(self.conv6.weight.data)
#         torch.nn.init.kaiming_uniform_(self.fc1.weight.data)
#         torch.nn.init.kaiming_uniform_(self.fc2.weight.data)
#
#
#     def forward(self, x, data1, data2):
#         # add sequence of convolutional and max pooling layers
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
#
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu3(x)
#         x = self.pool3(x)
#
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.relu4(x)
#         x = self.pool4(x)
#
#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = self.relu5(x)
#         x = self.pool5(x)
#
#
#         x = self.conv6(x)
#         x = self.bn6(x)
#         x = self.relu6(x)
#
#
#
#         x = x.view(-1,6272)
#
#         data1 = data1.reshape(-1, 1)
#         data2 = data1.reshape(-1, 1)
#
#         x = torch.cat((x, data1), dim=1)
#         x = torch.cat((x, data2), dim=1)
#
#
#         x = self.dropout7(x)  # 답작성
#         x = self.relu7(self.bcfc1(self.fc1(x)))
#         x = self.dropout8(x)  # 답작성
#         x = self.fc2(x)
#
#
#
#         return x

class ConvNet5(nn.Module):
    def __init__(self):
        super(ConvNet5, self).__init__()
        # convolutional layer (sees 3x224x224 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 16x112x112 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 32x56x56 tensor)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(2, 2)

        # convolutional layer (sees 32x28x28 tensor)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d(2, 2)  # 답작성)

        # 64x 14x14
        self.conv5 = nn.Conv2d(64,64, 3, padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(True)
        self.pool5 = nn.MaxPool2d(2, 2)  # 답작성)

        # 64x7x7
        self.conv6 = nn.Conv2d(64,128, 3, padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU(True)

        # 128x7x7
        self.dropout7 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(6272+2, 500)  # 답작성)
        self.bcfc1 = nn.BatchNorm1d(500)
        self.relu7 = nn.ReLU(True)
        # linear layer (500 -> 10)
        self.dropout8 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(500, 5)  # 답작성)




        self.initialize_weights()

    def initialize_weights(self):

        torch.nn.init.kaiming_uniform_(self.conv1.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv2.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv3.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv4.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv5.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv6.weight.data)
        torch.nn.init.kaiming_uniform_(self.fc1.weight.data)
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data)


    def forward(self, x, data1, data2):
        # add sequence of convolutional and max pooling layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)


        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)



        x = x.view(-1,6272)

        data1 = data1.reshape(-1, 1)
        data2 = data1.reshape(-1, 1)

        x = torch.cat((x, data1), dim=1)
        x = torch.cat((x, data2), dim=1)


        x = self.dropout7(x)  # 답작성
        x = self.relu7(self.bcfc1(self.fc1(x)))
        x = self.dropout8(x)  # 답작성
        x = self.fc2(x)



        return x


pred1 = []
li1 = []


# model1 = ConvNet()
# model1.load_state_dict(torch.load('C:/deep_basic/2023-final/convet_18.pt'))
# model1.to(device)
#

id_list = []
pred_list = []
#
# with torch.no_grad():
#     model1.eval()
#     for images, file_name, other in test_loader:
#         images = images.to(device)
#
#         other1 = other[0].to(device)
#         other2 = other[1].to(device)
#         logits1 = model1(images, other1, other2)
#
#
#
# pred2 = []
# li2 = []



# model2 = ConvNet2()
# # model2.load_state_dict(torch.load('C:/deep_basic/2023-final/convet_best_MODEL.pt'))
# model2.load_state_dict(torch.load('C:/deep_basic/2023-final/convet_best3_2_20_MODEL.pt'))
# model2.to(device)
# #
#
# with torch.no_grad():
#     model2.eval()
#     for images, file_name, other in test_loader:
#         images = images.to(device)
#
#         other1 = other[0].to(device)
#         other2 = other[1].to(device)
#         logits2 = model2(images, other1, other2)


model2 = ConvNet5()#63.7
# # model2.load_state_dict(torch.load('C:/deep_basic/2023-final/convet_best_MODEL.pt'))
model2.load_state_dict(torch.load('C:/deep_basic/2023-final/6_19_8_49/convet_best3_6_19_8_49_1_MODEL.pt'))
model2.to(device)


# with torch.no_grad():
#     model2.eval()
#     for images, file_name, other in test_loader:
#         images = images.to(device)
#
#         other1 = other[0].to(device)
#         other2 = other[1].to(device)
#         logits2 = model2(images, other1, other2)

model5 = ConvNet5()#최고 성능 64
model5.load_state_dict(torch.load('C:/deep_basic/2023-final/convet_best3_6_19_11_36_1_MODEL.pt'))
model5.to(device)

pred3 = []
li3 = []

model3 = ConvNet3()#62.773
# # model3.load_state_dict(torch.load('C:/deep_basic/2023-final/convet_best1_MODEL.pt'))
# model3.load_state_dict(torch.load('C:/deep_basic/2023-final/convet_best3_1_54_MODEL.pt'))
model3.load_state_dict(torch.load('C:/deep_basic/2023-final/convet_best3_6_2011512839_MODEL.pt'))
model3.to(device)


#
# with torch.no_grad():
#     model3.eval()
#     for images, file_name, other in test_loader:
#         images = images.to(device)
#
#         other1 = other[0].to(device)
#         other2 = other[1].to(device)
#         logits3 = model3(images, other1, other2)



model4 = ConvNet4()
model4.load_state_dict(torch.load('C:/deep_basic/2023-final/convet_best3_6_19_MODEL.pt'))
model4.to(device)

# with torch.no_grad():
#     model4.eval()
#     for images, file_name, other in test_loader:
#         images = images.to(device)
#
#         other1 = other[0].to(device)
#         other2 = other[1].to(device)
#         logits4 = model4(images, other1, other2)

with torch.no_grad():
    # model1.eval()
    model2.eval()#각각의 모델 평가
    model3.eval()
    model4.eval()
    model5.eval()

    for images, file_name, other in test_loader:
        images = images.to(device)

        other1 = other[0].to(device)
        other2 = other[1].to(device)

    #앙상블 가중치
        # weight2 = 0.3
        # weight3 = 0.1
        # weight4 = 0.2
        # weight5 = 0.4
#
        # logits1 = model1(images, other1, other2)
        logits2 = model2(images, other1, other2)#ensemble을 하기 위하여 각 모델들의 예측값을 가지고와서
        logits3 = model3(images, other1, other2)
        logits4 = model4(images, other1, other2)
        logits5 = model5(images, other1, other2)

        #logits =  (weight2*logits2) + (weight3*logits3) + (weight4*logits4) + (weight5*logits5)

        logits = logits2 + logits3  + logits4  + logits5#각 모델의 예측 값을 모두 더해서
        ps = F.softmax(logits, dim=1)#확률로

        top_p, top_class = ps.topk(1, dim=1)

        id_list += list(file_name)
        pred_list += top_class.T.tolist()[0]

handout_result = pd.DataFrame({'Id': id_list, 'Category': pred_list})
handout_result.to_csv('./ensemble_6_20_12_51_5858_최종.csv', index=False)
