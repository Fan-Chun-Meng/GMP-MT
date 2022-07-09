#  coding:utf-8
import time
import torch
import numpy as np

import res_self_attention

# from model import model, loss_func, CUDA, optimizer
from Dataset.Dataset import MyDataset

from regular import Regularization

st = time.time()
# model = DenseNet(Bottleneck,[2, 5, 4, 6]).to("cuda")
model = torch.load("model_useHdf5.pkl")#res_self_attention.ResLstm(3).cuda()

CUDA_LAUNCH_BLOCKING="1"
# model = torch.load("model-30s.pkl")
# 获取训练集与测试集以 8:2 分割
f = open(r"loss_PGA_useHdf5.txt","a")

lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
reg_loss = Regularization(model, 0.001, p=1).to("cuda")
device = "cuda"
trainloader = torch.utils.data.DataLoader(MyDataset(dataPath='tool/labelG.txt'), batch_size=64, shuffle=True)  # 训练集
trainloader_test = torch.utils.data.DataLoader(MyDataset(dataPath='tool/labelTG.txt'), batch_size=64, shuffle=True)  # 训练集
totalTestAvg = 10000
count = 10000
losslog = []
avevalue = []
def mse(output,targets):
    num = 0
    for i in range(len(output)):
        target = targets[i]
        if target > 3 and target < 4:
            num += 0.21*(target-output[i])*(target-output[i])
        if target >= 4 and target < 5:
            num += 0.32*(target-output[i])*(target-output[i])
        if target >= 5:
            num += 0.58*(target-output[i])*(target-output[i])
        if target >= 0 and target < 1:
            num += 0.33*(target-output[i])*(target-output[i])
        if target >= 1 and target < 2:
            num += 0.19*(target-output[i])*(target-output[i])
        if target >= 2 and target < 3:
            num += 0.18*(target-output[i])*(target-output[i])
    return num/len(output)

def TestModel():
    model_test = torch.load("model_useHdf5.pkl")
    global trainloader_test, totalTestAvg

    # model.train()  # enter train mode
    # print('\nEpoch: %d' % epoch)
    train_loss = 0  # accumulate every batch loss in a epoch
    correct = 0  # count when model' prediction is correct i train set
    total = 0  # total number of prediction in train set
    total = []
    avg2 = []
    avg = []
    avg1 = []
    for batch_idx, (inputs, dataSet_Nor, targets) in enumerate(trainloader_test):
        inputs, dataSet_Nor, targets = inputs.to(device), dataSet_Nor.to(device), targets.to(
            "cuda")  # load data to gpu device
        outputs = model_test(inputs.type(torch.FloatTensor).to(device), dataSet_Nor.type(torch.FloatTensor).to(device))
        # outputs = torch.abs(outputs)

        # optimizer.zero_grad()
        # loss.requires_grad = True
        # loss.backward()
        # optimizer.step()
        outputs = outputs.squeeze(1)
        # optimizer.zero_grad()  # clear gradients of all optimized torch.Tensors'
        loss = mse(outputs, targets)  # compute loss
        # loss.backward()  # compute gradient of loss over parameters
        # optimizer.step()  # update parameters with gradient descent
        train_loss += loss.item()  # accumulate every batch loss in a epoch
        # for t in range(len(targets)):
        #     if outputs[t] == 0:
        #         avg1.append(torch.log10(targets[t]))
        #     else:
        avg1.append(targets-outputs)

        if True:
            avg1 = [j.detach().cpu().numpy() for j in avg1]
            avg1 = list(map(abs, avg1))
            fin = np.mean(avg1)
            avg2 = [i * i for i in avg1]
            total.append(fin)
            # total.append(fin)
            a = str(np.max(avg1[0]))
            # maxlist.append(a)
            b = str(np.min(avg1[0]))


            # minlist.append(b)
            print(
                "Test loss: " + str((train_loss / (batch_idx + 1))) + " | avg : " + str(fin) + " (" + a + "/" + b + ")")
            f.write("Test loss: " + str((train_loss / (batch_idx + 1))) + " | avg : " + str(fin) + " (" + a + "/" + b + ")\n")

            avg1 = []

    if totalTestAvg > np.mean(total):
        totalTestAvg = np.mean(total)
        torch.save(model_test, 'model.pkl')
    print("total testAvg:" + str(np.mean(total)))
    f.write("total testAvg:" + str(np.mean(total))+"\n")

def TrainModel():
    for epoch in range(500):
        print('\nEpoch: %d' % epoch)
        model.train()  # enter train mode
        if epoch % 5 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.8
        train_loss = 0  # accumulate every batch loss in a epoch
        total = []
        avg2 = []
        avg = []
        for batch_idx, (inputs, dataSet_Nor, targets) in enumerate(trainloader):

            inputs, dataSet_Nor, targets  = inputs.to(device), dataSet_Nor.to(device), targets.to(device)  # load data to gpu device
            outputs = model(inputs.type(torch.FloatTensor).to(device), dataSet_Nor.type(torch.FloatTensor).to(device))
            outputs = outputs.squeeze(1)

            optimizer.zero_grad()  # clear gradients of all optimized torch.Tensors'
            loss = mse(outputs, targets)
            loss += reg_loss(model)
            loss.backward()  # compute gradient of loss over parameters
            optimizer.step()  # update parameters with gradient descent
            train_loss += loss.item()  # accumulate every batch loss in a epoch

            losslog.append(loss.item())
            avg1 = []


            avg1.append(targets-outputs)
            avg1 = [j.detach().cpu().numpy() for j in avg1]
            for i in range(len(avg1[0])):
                avg.append(avg1[0][i])
            avevalue.append(np.mean(avg))
            # print loss and acc
            if (batch_idx%10)==0:
                avg = list(map(abs, avg))
                fin = np.mean(avg)
                avg2 = [i*i for i in avg]
                total.append(fin)
                a = str(np.max(avg))
                # maxlist.append(a)
                b = str(np.min(avg))
                avg = []
                # minlist.append(b)
                trainLoss = train_loss / (batch_idx + 1)
                train_loss = 0
                print("Train loss: "+str((trainLoss))+" | avg_error : "+str(fin)+" (max:"+a+"/min:"+b+")")
                f.write("Train loss: "+str((trainLoss))+" | avg_error : "+str(avg)+" (max:"+a+"/min:"+b+")\n")
        # if (epoch % 50) == 0:
        #     x1 = range(0, 200)
        #     x2 = range(0, 200)
        #     y1 = avevalue
        #     y2 = losslog
        #     plt.subplot(2, 1, 1)
        #     plt.plot(x1, y1, 'o-')
        #     plt.title('Test accuracy vs. epoches')
        #     plt.ylabel('Test accuracy')
        #     plt.subplot(2, 1, 2)
        #     plt.plot(x2, y2, '.-')
        #     plt.xlabel('Test loss vs. epoches')
        #     plt.ylabel('Test loss')
        #     plt.show()TestModel()


        count = np.mean(total)
        print("total avg:" + str(np.mean(total))+"|std:"+str(sum(avg2)))
        f.write(
            "total avg:" + str(np.mean(total))+"|std:"+str(sum(avg2))+"\n")
        torch.save(model, 'model_useHdf5.pkl')
        TestModel()

TrainModel()
