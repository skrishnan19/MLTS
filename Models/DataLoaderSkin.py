import csv
import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import random
import torch
from torchvision import datasets, transforms
# from torchvision.transforms import v2
import copy
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms import AutoAugment



def printStats(y):
    uniqueLbls = np.unique(y)
    for lbl in uniqueLbls:
        idx = np.sum(y == lbl)
        print(f'{lbl:1d}', " : ", f'{idx:7d}')
    print(f'{"total":7s}', " : ", f'{len(y):7d}')

def calWeights(lbl_arr):
    weights = []
    unique_lbls = np.unique(lbl_arr)
    for lbl in unique_lbls:
        idx = [i for i, x in enumerate(lbl_arr) if x == lbl]
        weights.append(1/len(idx))
    weights = np.asarray(weights)
    weights = weights/sum(weights)
    return weights

def pil_loader(path):
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')

def getTransforms():
    imsize1 = 280
    imsize2 = 224
    normalize = transforms.Normalize(mean=[0.6745, 0.4796, 0.5008], std=[0.2533, 0.2059, 0.2198])

    tw = transforms.Compose([
        transforms.Resize(imsize1),
        transforms.CenterCrop([imsize2, imsize2]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation([-180, 180]),
        transforms.ToTensor(),
        normalize
    ])

    ts = transforms.Compose([
        transforms.Resize(imsize1),
        transforms.RandomCrop([imsize2, imsize2]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=(0.2), contrast=(0.2)),
        transforms.RandomAffine(degrees=[-180, 180], scale=[0.9, 1.1]), 
        transforms.ToTensor(),
        normalize,
    ])

    tt = transforms.Compose([
        transforms.Resize(imsize1),
        transforms.CenterCrop([imsize2, imsize2]),
        transforms.ToTensor(),
        normalize
    ])
    return tw, ts, tt

def getData(dataset, trainTestVal, itrNo):
    if dataset == 'ISIC2018':
        if trainTestVal == 'train':
            tmpFn = 'training.csv'
        elif trainTestVal == 'test':
            tmpFn = 'testing.csv'
        elif trainTestVal == 'val':
            tmpFn = 'validation.csv'
        return getData_2018(tmpFn)
    elif dataset == 'ISIC2019':
        if trainTestVal == 'train':
            tmpFn = 'training_' + str(itrNo) + '.csv'
        elif trainTestVal == 'test':
            tmpFn = 'testing_' + str(itrNo) + '.csv'
        elif trainTestVal == 'val':
            tmpFn = 'validation_' + str(itrNo) + '.csv'
        return getData_2019(tmpFn)

def getData_2019(csv_file):
    
    tmp = '/storage/scratch1/phd23-pg-skin-classification/MLTS/DATA/2019/Splits/'
    file = pd.read_csv(os.path.join(tmp, csv_file))
    filenames = file['filename'].values
    labels = file['lbl'].values.astype(int)
    lblArr = []
    for i, row in enumerate(labels):
        lblArr.append(row)

    dirName = "/storage/scratch1/phd23-pg-skin-classification/ISIC2019/ISIC_2019_Training_Input/"
    filenames = [os.path.join(dirName, fn + '.jpg' ) for fn in filenames]
    return filenames, lblArr

def getData_2018(csv_file):
    tmp = "/storage/scratch1/phd23-pg-skin-classification/MLTS/DATA/2018/Splits/"
    file = pd.read_csv(os.path.join(tmp, csv_file))
    filenames = file['image'].values
    labels = file.iloc[:, 1:-1].values.astype(int)
    lblArr = []
    for i, row in enumerate(labels):
        filenames[i] = filenames[i] + '.jpg'
        for k in range(len(row)):
            if row[k] == 1:
                lblArr.append(k)
                break
    dirName = "/storage/scratch1/phd23-pg-skin-classification/mydatasets/ISIC2018_Task3_Training_Input/"
    filenames = [os.path.join(dirName, fn) for fn in filenames]
    return filenames, lblArr

def splitData_Pecentage(lblArr, pL):
    uniqueLbls = np.unique(lblArr)
    trainL_idx = []
    trainUL_idx = []
    for lbl in uniqueLbls:
        idx = [i for i, x in enumerate(lblArr) if x==lbl]
        idx = random.sample(idx, len(idx))

        # labeled train
        nl = int(np.ceil(len(idx) * pL))
        tmpL = idx[0:nl]
        trainL_idx.extend(tmpL)

        # unlabeled train
        if pL != 1:
            tmpUL = idx[nl:]
            trainUL_idx.extend(tmpUL)
    return trainL_idx, trainUL_idx


def getDataLoaders(dataset, itrNo, seedval, pL, addValDataWithTrain, bs_l, bs_u):
    if dataset == 'ISIC2018' or dataset == 'ISIC2019':
        labeled_trainloader, unlabeled_trainloader, test_loader = getDataLoaders_SKIN(dataset, itrNo, seedval, pL, addValDataWithTrain, bs_l, bs_u)
    printStatDataloaders(labeled_trainloader, unlabeled_trainloader, test_loader)
    return labeled_trainloader, unlabeled_trainloader, test_loader

def getDataLoaders_SKIN(dataset, itrNo, seedval, pL, addValDataWithTrain, bs_l, bs_u):
    random.seed(seedval)
    fnArr, lblArr = getData(dataset, 'train', itrNo)

    if addValDataWithTrain:
        print('ntrain = ', len(lblArr))
        print('Adding validation data to the training data')
        fnArr_v, lblArr_v = getData(dataset, 'val', itrNo)
        print('nval = ', len(lblArr_v))
        lblArr = lblArr + lblArr_v
        fnArr = fnArr + fnArr_v
        print('ntrain + nval = ', len(lblArr))
    fnArr = np.array(fnArr)
    lblArr = np.array(lblArr)

    trainL_idx, trainUL_idx = splitData_Pecentage(lblArr, pL)
    cw = calWeights(lblArr[trainL_idx])
    print(cw)

    num_workers = 16
    dataset_L = DatasetSkin(False, fnArr[trainL_idx], lblArr[trainL_idx])
    labeled_trainloader = DataLoader(
        dataset_L,
        shuffle=True,
        batch_size=bs_l,
        num_workers=num_workers,
        drop_last=True)
    print('BS (L, UL) = (', bs_l, bs_u, ')')

    unlabeled_trainloader = None
    if pL != 1:
        dataset_UL = DatasetSkin(False, fnArr[trainUL_idx], lblArr[trainUL_idx])
        unlabeled_trainloader = DataLoader(
            dataset_UL,
            shuffle=True,
            batch_size=bs_u,
            num_workers=num_workers,
            drop_last=True)

    fnArr, lblArr = getData(dataset, 'test', itrNo)
    dataset_test = DatasetSkin(True, fnArr, lblArr)
    test_loader = DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=bs_l,
        num_workers=num_workers)

    return labeled_trainloader, unlabeled_trainloader, test_loader #, cw, np.unique(lblArr)

def printStatDataloaders(labeled_trainloader, unlabeled_trainloader, test_loader):
    y_L, y_te = labeled_trainloader.dataset.lblArr, test_loader.dataset.lblArr
    y_UL = None
    if unlabeled_trainloader is not None:
        y_UL = unlabeled_trainloader.dataset.lblArr
    unique_lbls = set(y_L)

    def countLbls(lblArr, lbl):
        idx = [i for i, x in enumerate(lblArr) if x == lbl]
        return len(idx)

    print('lbl \t Lbl \t UNLbl \t Test')
    for lbl in unique_lbls:
        tmp_L = countLbls(y_L, lbl)
        tmp_un = 0
        if y_UL is not None:
            tmp_un = countLbls(y_UL, lbl)
        tmp_te = countLbls(y_te, lbl)
        print('%1d\t%5d\t%5d\t%5d'%(lbl, tmp_L, tmp_un, tmp_te))
    if y_UL is not None:
        print('\t%5d\t%5d\t%5d' % (len(y_L), len(y_UL), len(y_te)))
    else:
        print('\t%5d\t%5d\t%5d' % (len(y_L), 0, len(y_te)))


class DatasetSkin(data.Dataset):
    def __init__(self, isTest, fnArr, lblArr):
        self.tw, self.ts, self.tt = getTransforms()
        self.isTest = isTest
        self.fnArr = fnArr
        self.lblArr = lblArr
        self.unique_lbls = np.unique(self.lblArr)
        self.data_len = len(self.lblArr)

    def __getitem__(self, idx):
        img_fn = self.fnArr[idx]
        I = pil_loader(img_fn)
        if self.isTest:
            return self.tt(I), self.lblArr[idx]
        else:
            return self.tw(I), self.ts(I), self.lblArr[idx], idx

    def __len__(self):
        return len(self.lblArr)

    
 