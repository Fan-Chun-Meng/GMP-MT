import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResLstm1(nn.Module):
    def __init__(self, in_planes, stride=1):
        super(ResLstm, self).__init__()
        self.inplanes = in_planes
        self.conv8_11 = nn.Sequential(
            nn.Conv1d(in_channels=in_planes, out_channels=8, kernel_size=11, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.conv16_9 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=9, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.conv16_7 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=7, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.conv32_7 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.conv32_5 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.conv64_5 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv64_3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.resconv64_3 =  self._make_layer(BasicBlock, 64, 2)
        self.resconv64_2 = self._make_layer(BasicBlock, 64, 2)
        self.bilstm = nn.LSTM(input_size=242, hidden_size=256, num_layers=1, bidirectional=False, batch_first=True)
        self.bilstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, bidirectional=False, batch_first=True)
        self.convTrans1 = nn.ConvTranspose1d(in_channels=64,kernel_size=3,stride=1,out_channels=64)
        self.convTrans2 = nn.ConvTranspose1d(in_channels=64, kernel_size=3, stride=1, out_channels=32)
        self.convTrans3 = nn.ConvTranspose1d(in_channels=32, kernel_size=5, stride=1, out_channels=16)
        self.convTrans4 = nn.ConvTranspose1d(in_channels=16,kernel_size=7,stride=1,out_channels=8)
        self.convTrans5 = nn.ConvTranspose1d(in_channels=8, kernel_size=7, stride=2, out_channels=3)
        self.fc0 = nn.Sequential(
            nn.Linear(8192, 64),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(3171, 64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU()
        )
        # 由nn.Parameter定义的变量都为requires_grad=True状态
        self.weight_W1 = nn.Parameter(torch.Tensor(512, 512))
        self.weight_proj1 = nn.Parameter(torch.Tensor(512, 1))
        self.weight_W2 = nn.Parameter(torch.Tensor(512, 512))
        self.weight_proj2 = nn.Parameter(torch.Tensor(512, 1))
        nn.init.uniform_(self.weight_W1, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj1, -0.1, 0.1)
        nn.init.uniform_(self.weight_W2, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj2, -0.1, 0.1)


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = None
        downsample = None
        previous_dilation = 1
        self.groups = 1
        self.base_width = 64
        layers = []
        layers.append(block(64, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(64, planes))

        return nn.Sequential(*layers)



    def forward(self, input, inputNor):

        conv_input = self.conv8_11(input)
        conv_input = self.conv16_9(conv_input)
        conv_input = self.conv16_7(conv_input)
        conv_input = self.conv32_7(conv_input)
        conv_input = self.conv32_5(conv_input)
        conv_input = self.conv64_5(conv_input)
        conv_input = self.conv64_3(conv_input)
        conv_input = self.resconv64_3(conv_input)
        conv_input = self.resconv64_2(conv_input)
        out,(fc,hc) = self.bilstm(conv_input)
        # u = torch.tanh(torch.matmul(out, self.weight_W1))
        # att = torch.matmul(u, self.weight_proj1)
        # att_score = F.softmax(att, dim=1)
        # scored_x = out * att_score
        out, (fc, hc) = self.bilstm2(out)
        # u = torch.tanh(torch.matmul(out, self.weight_W2))
        # att = torch.matmul(u, self.weight_proj2)
        # att_score = F.softmax(att, dim=1)
        # scored_x = out * att_score
        # feat = torch.sum(scored_x, dim=1)
        # out = self.convTrans1(out)
        # out = self.convTrans2(out)
        # out = self.convTrans3(out)
        # out = self.convTrans4(out)
        # out = self.convTrans5(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc0(out)

        #out = self.fc(out)
        out = self.fc2(out)

        return out


