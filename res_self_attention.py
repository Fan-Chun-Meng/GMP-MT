import math

import torch
import torch.nn as nn
import torch.nn.functional as F
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        hidden_size = 242
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


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

class ResLstm(nn.Module):
    def __init__(self, in_planes, stride=1):
        super(ResLstm, self).__init__()
        self.inplanes = in_planes
        hidden_size = 244
        num_attention_heads = 1
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, padding=1, stride=1)

        self.key = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, padding=1, stride=1)
        self.value = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, padding=1, stride=1)

        self.attn_dropout = nn.Dropout(0.5)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, 242)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(0.5)

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
        self.fc0 = nn.Sequential(
            nn.Linear(15488, 1)
        )


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

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


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

        mixed_query_layer = self.query(conv_input)
        mixed_key_layer = self.key(conv_input)
        mixed_value_layer = self.value(conv_input)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + conv_input)

        view = hidden_states.view(len(hidden_states),-1)

        return self.fc0(view)


