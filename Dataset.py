from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import math
import numpy as np
from scipy import signal
class MyDataset(Dataset):

    def __init__(self, dataPath):
        dataPath = open(dataPath, 'r')
        data = dataPath.readlines()

        self.data = data

    def __getitem__(self, index):

        path = self.data[index]
        dataPath = path.split("\t")[2]
        label = float(path.split("\t")[0])
        dataSet = np.zeros((3, 1000))
        dataSet_PGV = np.zeros((3, 1000))
        dataPath1 = ''
        dataPath2 = ''

        dataPath1 = dataPath.replace("BHZ","BHN")
        dataPath2 = dataPath.replace("BHZ","BHE")
        f = open(dataPath.split("\n")[0],"r")
        f1 = open(dataPath1.split("\n")[0], "r")
        f2 = open(dataPath2.split("\n")[0], "r")
        if dataPath2 == " " or dataPath2 == "":
            print(path)

        l = list(map(int, f.readlines()[10000:11001]))
        a = []
        b = []
        for i in range(1,1001):
           b.append(l[i-1])
           a.append((l[i]-l[i-1])/0.01)
        cb, ca = signal.butter(3, [2 * 0.1 / 100, 2 * 10 / 100], 'bandstop')  # 配置滤波器 8 表示滤波器的阶数
        filtedDatar = signal.filtfilt(cb, ca, np.array(a))  # data为要过滤的信号
        filtedDatar_PGV = signal.filtfilt(cb, ca, np.array(b))  # data为要过滤的信号
        dataSet[0] = a
        dataSet_PGV[0] = filtedDatar_PGV
        l = list(map(int, f1.readlines()[10000:11001]))
        a = []
        b = []
        for i in range(1, 1001):
            b.append(l[i-1])
            a.append((l[i] - l[i - 1]) / 0.01)
        cb, ca = signal.butter(3, [2 * 0.1 / 100, 2 * 10 / 100], 'bandstop')  # 配置滤波器 8 表示滤波器的阶数
        filtedDatar = signal.filtfilt(cb, ca, np.array(a))  # data为要过滤的信号
        filtedDatar_PGV = signal.filtfilt(cb, ca, np.array(b))  # data为要过滤的信号
        dataSet[1] = a
        dataSet_PGV[1] = filtedDatar_PGV
        l = list(map(int, f2.readlines()[10000:11001]))
        a = []
        b = []
        for i in range(1, 1001):
            b.append(l[i-1])
            a.append((l[i] - l[i - 1]) / 0.01)
        cb, ca = signal.butter(3, [2 * 0.1 / 100, 2 * 10 / 100], 'bandstop')  # 配置滤波器 8 表示滤波器的阶数
        filtedDatar = signal.filtfilt(cb, ca, np.array(a))  # data为要过滤的信号
        filtedDatar_PGV = signal.filtfilt(cb, ca, np.array(b))  # data为要过滤的信号
        dataSet_PGV[2] = filtedDatar_PGV
        dataSet[2] = a
        # dataSet_Nor = dataSet - np.mean(dataSet,axis=1,keepdims=True)
        # dataSet = np.abs(dataSet)
        dataSet_Nor = (dataSet - np.min(dataSet, axis=1,keepdims=True)) / (np.max(dataSet, axis=1,keepdims=True) - np.min(dataSet, axis=1,keepdims=True) )
        # dataSet_Nor = (dataSet_Nor - np.mean(dataSet_Nor, axis=1,keepdims=True)) / np.std(dataSet_Nor, axis=1,keepdims=True)

        return torch.tensor(dataSet), torch.tensor(dataSet_Nor), torch.tensor(float(label))

    def __len__(self):
        return len(self.data)

# if __name__ == '__main__':
#
#
#     trainset = MyDataset(dataPath='train_I_PGA_5-1.txt')  # 训练集
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
#
#     for step, (tx, ty) in enumerate(trainloader):
#         print('---test---', tx, ty)
