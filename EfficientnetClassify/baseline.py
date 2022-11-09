import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data.dataset import Dataset, T_co
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import os
import sys
import glob
import numpy as np
from tqdm import tqdm
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
import time


err_tol = {
    'challenge': [0.30, 0.40, 0.50, 0.60, 0.70],
    'play': [0.15, 0.20, 0.25, 0.30, 0.35],
    'throwin': [0.15, 0.20, 0.25, 0.30, 0.35]
}

train_csv = pd.read_csv('kaggle-dfl-bundesliga-shootout/train.csv')
submission = pd.read_csv('kaggle-dfl-bundesliga-shootout/sample_submission.csv')


# 数据集类
class DFLDataset(Dataset):
    def __init__(self, img_path, img_label, transform):
        self.img_path = img_path
        self.img_label = img_label
        self.indices = range(len(img_path))
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        # index = self.indices[index]
        image = cv2.imread(self.img_path[index])
        image = image.astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image,torch.from_numpy(np.array(self.img_label[index]))

    def __len__(self):
        return len(self.img_path)


train_path = glob.glob('kaggle-dfl-bundesliga-shootout/split_images/train/*/*')
train_label = [x.split('\\')[-2] for x in train_path]

val_path = glob.glob('kaggle-dfl-bundesliga-shootout/split_images/val/*/*')
val_label = [x.split('\\')[-2] for x in val_path]

train_df = pd.DataFrame({'path': train_path,
                         'label': train_label})
val_df = pd.DataFrame({'path': val_path,
                       'label': val_label})

train_df['label_int'], lbl_train = pd.factorize(train_df['label'])
val_df['label_int'], lbl_val = pd.factorize(val_df['label'])
# train_df = train_df.sample(frac=1)

train_dataloader = DataLoader(DFLDataset(train_df['path'].values,
                                         train_df['label_int'].values,
                                         A.Compose([
                                             A.Resize(224, 224),
                                             A.RandomContrast(p=0.5),
                                             A.RandomBrightness(p=0.5),
                                             A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                             ToTensorV2()
                                         ])
                                         ),
                              batch_size=10, shuffle=True, num_workers=0, pin_memory=True)

val_dataloader = DataLoader(DFLDataset(val_df['path'].values,
                                       val_df['label_int'].values,
                                       A.Compose([
                                           A.Resize(224, 224),
                                           A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                           ToTensorV2()
                                       ])
                                       ),
                            batch_size=10, shuffle=True, num_workers=0, pin_memory=True)


def train(train_loader, model, criterion,optimizer, epoch):
    model.train()
    train_loss = 0
    for (input_, target) in tqdm(train_loader, desc='train epoch '+str(epoch)):
        input_ = input_.cuda()
        target = target.cuda()

        output_ = model(input_)
        loss = criterion(output_, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)

def validate(val_loader, model, criterion,epoch):
    model.eval()
    val_acc = 0

    with torch.no_grad():
        for (input_, target) in tqdm(val_loader, desc='validate epoch '+str(epoch)):
            input_ = input_.cuda()
            target = target.cuda()

            output_ = model(input_)
            loss = criterion(output_, target)

            val_acc += (output_.argmax(1) == target).sum().item()
    return val_acc / len(val_loader.dataset)

def predict(test_loader, model, criterion):
    model.eval()
    test_pred = []
    with torch.no_grad():
        end = time.time()
        for i, (input_, target) in enumerate(test_loader):
            input_ = input_.cuda()
            target = target.cuda()

            output = model(input_)
            test_pred.append(output.data.cpu().numpy())

    return np.vstack(test_pred)


from efficientnet_pytorch.model import EfficientNet
model = EfficientNet.from_pretrained(model_name='efficientnet-b1',num_classes=4)
model = model.to('cuda')
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(params=model.parameters(), lr=0.005)
scheduler = torch.torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=5, mode="triangular2")

best_acc = 0
for i in range(5):
    train_loss = train(train_loader=train_dataloader, model=model, criterion=criterion, optimizer=optimizer, epoch=i)
    val_acc = validate(val_loader=val_dataloader, model=model, criterion=criterion,epoch=i)
    if val_acc > best_acc:
        torch.save(model.state_dict(), 'model.pth')
        best_acc = val_acc

    scheduler.step()
    print(train_loss, val_acc)

