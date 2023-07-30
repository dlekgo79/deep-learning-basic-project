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

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#
# set_seed(42)

train_dataframe = pd.read_csv('C:/deep_basic/2023-final/data.csv')#train data 블러오기
test_df = pd.read_csv('C:/deep_basic/2023-final/testdata.csv')#test data 불러오기

from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(train_dataframe, shuffle=True, test_size=0.15, stratify=train_dataframe['Label'])
#train과 validation set 나누기


# custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, train='train', transform=None):
        if train == 'train':# train의 경우
            self.image_list = []#image list 생성
            self.label_list = []#label list 생성
            self.other_list = []#인종 나이 성별 리스트 생성
            path = 'C:/deep_basic/2023-final/dataset/{}/{}'
            for index, row in dataframe.iterrows():
                image_path = row['Image'] #데이터 path 를 불러오기
                image_label = row['Label']#데이터 path 불러오기
                image_age = row['Age']#나이 path
                image_gender = row['Gender']#성별 path
                image_race = row['Race']#나이 path
                image = Image.open(path.format(image_label, image_path)).convert('RGB')#경로를 통해서 이미지 가지고 오기
                # if there is transform, apply transform
                if transform != None:
                    image = transform(image)
                self.image_list.append(image)#만들어 놓은 리스트에 추가
                self.label_list.append(image_label)#만들어 놓은 리스트에 추가
                self.other_list.append((image_age, image_gender, image_race))#만들어 놓은 리스트에 추가

        elif train == 'test':#test의 경우
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

train_transform = transforms.Compose([#train transform

    transforms.RandomHorizontalFlip(),# train 이미지 좌우 반전
    transforms.Resize(224),# train 이미지 크기 변경
    transforms.ToTensor(),# train 이미지 ToTensor()로 변환해서 학습 할 수 있게!
    transforms.Normalize([0.602927 ,0.46158618 ,0.39498535], [0.21958497, 0.19605462 ,0.18665607])# train 이미지의 평균과 표준 편차로 나누어 정규화 하는 것

])

test_transform = transforms.Compose([#test transform
    transforms.Resize(224),#test 이미지 크기 변경
    transforms.ToTensor(),#test 이미지 ToTensor()로 변경
    transforms.Normalize([0.602927 ,0.46158618, 0.39498535],[0.21958497 ,0.19605462, 0.18665607])#test 이미지 평균과 표준 편차로 나누어 정규화
])

train_dataset = CustomDataset(train_df, train='train', transform=train_transform)
valid_dataset = CustomDataset(valid_df, train='train', transform=test_transform)
test_dataset = CustomDataset(test_df, train='test', transform=test_transform)

# dataset에 대한 data loaders 구성

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# normalize를 위해 rgb 채널의 mean, std 값 구하기

images,labels,other = next(iter(train_loader))
#
# meanRGB = [np.mean(x.numpy(), axis=(2,3)) for x,label,other in train_loader]
# stdRGB = [np.std(x.numpy(), axis=(2,3)) for x,label,other in train_loader]
#
# meanR = np.mean([m[0] for m in meanRGB])
# meanG = np.mean([m[1] for m in meanRGB])
# meanB = np.mean([m[2] for m in meanRGB])
#
# stdR = np.mean([s[0] for s in stdRGB])
# stdG = np.mean([s[1] for s in stdRGB])
# stdB = np.mean([s[2] for s in stdRGB])
#
# print(meanR, meanG, meanB)
# print(stdR, stdG, stdB)
# # RGB별로 평균,표준 편차




# model
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # convolutional layer (sees 3x224x224) -> 이미지를 resize 했으므로 224 x 224 x 3(channel)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=(1, 1))# 1층의 CONV2d은 in_channel이 3 output_channel이 16 padding은 1 stride 1 kernel 3
        self.bn1 = nn.BatchNorm2d(16) # 1층의 Batchnorm은 out_channel의 크기 16
        self.relu1 = nn.ReLU(True)#activation function은 relu
        self.pool1 = nn.MaxPool2d(2, 2)#Maxpool을 통해서 이미지의 크기를 절반으로 줄임

        # convolutional layer (sees 16x112x112)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=(1, 1))# 2층의 CONV2d은 in_channel이 16 output_channel이 32 padding은 1 stride 1 kernel 3
        self.bn2 = nn.BatchNorm2d(32)# 2층의 Batchnorm은 out_channel의 크기 32
        self.relu2 = nn.ReLU(True)#activation function은 relu
        self.pool2 = nn.MaxPool2d(2, 2)#Maxpool을 통해서 이미지의 크기를 절반으로 줄임

        # convolutional layer (sees 32x56x56)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=(1, 1))#3층의 CONV2d은 in_channel이 32 output_channel이 32 padding은 1 stride 1 kernel 3
        self.bn3 = nn.BatchNorm2d(32)# 3층의 Batchnorm은 out_channel의 크기 32
        self.relu3 = nn.ReLU(True)#activation function은 relu
        self.pool3 = nn.MaxPool2d(2, 2)#Maxpool을 통해서 이미지의 크기를 절반으로 줄임

        # convolutional layer (sees 32x28x28)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=(1, 1))#4층의 CONV2d은 in_channel이 32 output_channel이 64 padding은 1 stride 1 kernel 3
        self.bn4 = nn.BatchNorm2d(64)# 4층의 Batchnorm은 out_channel의 크기 64
        self.relu4 = nn.ReLU(True)#activation function은 relu
        self.pool4 = nn.MaxPool2d(2, 2)#Maxpool을 통해서 이미지의 크기를 절반으로 줄임

        # 64x 14x14
        self.conv5 = nn.Conv2d(64,64, 3, padding=(1, 1))#5층의 CONV2d은 in_channel이 64 output_channel이 64 padding은 1 stride 1 kernel 3
        self.bn5 = nn.BatchNorm2d(64)# 5층의 Batchnorm은 out_channel의 크기 64
        self.relu5 = nn.ReLU(True)#activation function은 relu
        self.pool5 = nn.MaxPool2d(2, 2)#Maxpool을 통해서 이미지의 크기를 절반으로 줄임

        # 64x7x7
        self.conv6 = nn.Conv2d(64,128, 3, padding=(1, 1))#6층의 CONV2d은 in_channel이 64 output_channel이 64 padding은 1 stride 1 kernel 3
        self.bn6 = nn.BatchNorm2d(128)# 6층의 Batchnorm은 out_channel의 크기 64
        self.relu6 = nn.ReLU(True)#activation function은 relu
        #특성 추출 끝

        #분류 시작
        # 128x7x7
        self.dropout7 = nn.Dropout(0.3)#뉴런을 무작위로 0.3 만큼 삭제하고 학습
        self.fc1 = nn.Linear(6272+2, 500)#conv6층에서 128 x 7 x 7 = 6272 그리고 인종 나이 특성 2개 추가
        self.bcfc1 = nn.BatchNorm1d(500)#Flatten 해서 1차원이 되어서 BatchNorm1d로 fc1의 out_channel 500이 입력으로
        self.relu7 = nn.ReLU(True)#activation function Relu
        # linear layer (500 -> 5)
        self.dropout8 = nn.Dropout(0.3)#뉴런을 무작위로 0.3 만큼 삭제하고 학습
        self.fc2 = nn.Linear(500, 5)#총 5 label로 분류를 해야하므로 out은 5

        self.initialize_weights()#가중치 초기화

    def initialize_weights(self):#각 layer 별로 가중치 초기화 -> Relu에 맞는 가중치 초기화 적용
        torch.nn.init.kaiming_uniform_(self.conv1.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv2.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv3.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv4.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv5.weight.data)
        torch.nn.init.kaiming_uniform_(self.conv6.weight.data)
        torch.nn.init.kaiming_uniform_(self.fc1.weight.data)
        torch.nn.init.kaiming_uniform_(self.fc2.weight.data)


    def forward(self, x, data1, data2):
        # add sequence of convolutional and max pooling layers #위의 구조와 같이 모델을 쌓아주었다.
        x = self.conv1(x)#conv2d()
        x = self.bn1(x)#batchnorm
        x = self.relu1(x)#activation funcrion
        x = self.pool1(x)#maxpooling

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


        x = x.view(-1,6272) #flatten을 해서 1차원으로 만들어서 분류하기 위해

        #특성 2개를 reshape를 통해서 batch_size 만큼의 열 뒤에 2개 씩 추가
        data1 = data1.reshape(-1, 1)#나이
        data2 = data1.reshape(-1, 1)#인종
        #열 뒤에 특성 2개를  합치기 위한 코드
        x = torch.cat((x, data1), dim=1)
        x = torch.cat((x, data2), dim=1)

        x = self.dropout7(x)
        x = self.relu7(self.bcfc1(self.fc1(x)))
        x = self.dropout8(x)
        x = self.fc2(x)



        return x


# create a complete CNN
model = ConvNet()
print(model)
# move tensors to GPU if CUDA is available
model.to(device)

# create a complete CNN

# move tensors to GPU if CUDA is available

# cost 함수 및 optim 설정
import torch.optim as optim

#label smoothing

nSamples = [1039,1032,1023,1023,1021]#각 클래스의 이미지의 개수
normedWeights = [1-(x/sum(nSamples)) for x in nSamples]#평균으로 나누어 정규화한 가중치 저장
normedWeights = torch.FloatTensor(normedWeights).to(device)#정규화된 가중치 뱐환
normedWeights = normedWeights/5
print(normedWeights)
weights = torch.FloatTensor([0.1596, 0.1598, 0.1602, 0.1602, 0.1603]).cuda()#가중치 값을 GPU로 올리기

criterion = nn.CrossEntropyLoss(weight =weights,label_smoothing=0.01)#label_smoothing을 통해 정도를 넣어줌

# criterion = nn.CrossEntropyLoss()



optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.05)  #optimizer


scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=0)#스케줄러
#
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Train and validation
# number of epochs to train the model
n_epochs = 50  # 65번은 오버피팅이 남

valid_loss_min = np.Inf  # track change in validation loss

# keep track of training and validation loss
train_loss = torch.zeros(n_epochs)#각 loss들 초기화
valid_loss = torch.zeros(n_epochs)

train_acc = torch.zeros(n_epochs)

valid_acc = torch.zeros(n_epochs)

for e in range(0, n_epochs):#학습 시작

    ###################
    # train the model #
    ###################
    model.train()#train 시작
    # Image,Label,Age,Gender,Race
    for images, labels, other in train_loader:
        # move tensors to GPU if CUDA is available
        images, labels, other1, other2 = images.to(device), labels.to(device), other[1].to(device), other[2].to(device)

        # clear the gradients of all optimized variables

        optimizer.zero_grad() #모든 gradient를 초기화
        # forward pass: compute predicted outputs by passing inputs to the model
        logits = model(images, other1, other2) #확률값 -> forward
        # calculate the batch loss


        loss = criterion(logits, labels)#위에서 구한 예측값을 넣어서 예측
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward() #backpropagation 과정
        optimizer.step()

        # perform a single optimization step (parameter update)
#
        # update training loss
        #train_loss[e] += loss.item()
        train_loss[e] += loss.detach().cpu().item()


        ps = F.softmax(logits, dim=1)#확률로 변환
        top_p, top_class = ps.topk(1, dim=1)#가장 확률이 높은 레이블
        equals = top_class == labels.reshape(top_class.shape)
        train_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()

    scheduler.step()#스케줄러를 update -> 정한 epoch마다
    train_loss[e] /= len(train_loader)
    train_acc[e] /= len(train_loader)
    # scheduler.step(train_loss[e])
    ######################
    # validate the model #
    ######################
    with torch.no_grad():
        model.eval() #검증
        for images, labels, other in valid_loader:
            # move tensors to GPU if CUDA is available
            images, labels, other1, other2 = images.to(device), labels.to(device), other[1].to(device), other[2].to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            logits = model(images, other1, other2)#validation data를 예측
            # calculate the batch loss
            loss = criterion(logits, labels)#정답값과 비교
            # update average validation loss
            valid_loss[e] += loss.item()

            ps = F.softmax(logits, dim=1)#이 예측값을 확률값으로
            top_p, top_class = ps.topk(1, dim=1)#가장 높은 레이블
            equals = top_class == labels.reshape(top_class.shape)
            valid_acc[e] += torch.mean(equals.type(torch.float)).detach().cpu()

    # calculate average losses
    # scheduler.step(valid_loss[e])
    valid_loss[e] /= len(valid_loader)
    valid_acc[e] /= len(valid_loader)


    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        e, train_loss[e], valid_loss[e]))

    # print training/validation statistics
    print('Epoch: {} \tTraining accuracy: {:.6f} \tValidation accuracy: {:.6f}'.format(
        e, train_acc[e], valid_acc[e]))

    # save model if validation loss has decreased
    if valid_loss[e] <= valid_loss_min:
        print('Validation loss decreased ({:.6f} '
              '--> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss[e]))
        # torch.save(model.state_dict(), 'convet_best3_6_19_11_48_1_MODEL.pt')
        valid_loss_min = valid_loss[e]

# LOSS 그래프
# model.load_state_dict(torch.load('convet_best3_6_19_11_48_1_MODEL.pt'))
import matplotlib.pyplot as plt

# loss graph
plt.plot(train_loss, label='training loss')
plt.plot(valid_loss, label='validation loss')
plt.legend()
plt.show()
# accuracy graph
plt.plot(train_acc, label = 'training accuracy')
plt.plot(valid_acc, label = 'validation accuracy')
plt.legend()
plt.show()
# test loss

id_list = []
pred_list = []

with torch.no_grad():
    model.eval()
    for images, file_name, other in test_loader:
        images = images.to(device)

        other1 = other[0].to(device)
        other2 = other[1].to(device)
        logits = model(images, other1, other2)

        ps = F.softmax(logits, dim=1)

        top_p, top_class = ps.topk(1, dim=1)

        id_list += list(file_name)
        pred_list += top_class.T.tolist()[0]

# handout_result = pd.DataFrame({'Id': id_list, 'Category': pred_list})
# handout_result.to_csv('./result_BEST3_MODEL_6_19_11_48_1.csv', index=False)



#matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

classes = ['1~10', '11~20', '21~30', '31~40', '41~']
classes_cm = [0, 1, 2, 3, 4]
test_loss = 0
y_pred = []
y_true = []
test_acc = 0
with torch.no_grad():
    model.eval()
    for data, labels, other in valid_loader:
        # move tensors to GPU if CUDA is available
        data, labels, other1, other2 = data.to(device), labels.to(device), other[1].to(device), other[2].to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        logits = model(data, other1, other2)
        # calculate the batch loss
        loss = criterion(logits, labels)
        # update average test loss
        test_loss += loss.item()

        top_p, top_class = logits.topk(1, dim=1)
        y_pred.extend(top_class.data.cpu().numpy())
        y_true.extend(labels.data.cpu().numpy())
        equals = top_class == labels.reshape(top_class.shape)
        test_acc += torch.sum(equals.type(torch.float)).detach().cpu()

    test_acc /= len(valid_loader.dataset)
    test_acc *= 100

cm = confusion_matrix(y_true, y_pred, labels=classes_cm, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot()
plt.show()
print('Test accuracy : {}'.format(test_acc))

