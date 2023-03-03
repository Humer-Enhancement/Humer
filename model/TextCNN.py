# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, config, pretrained_weights):
        super(TextCNN, self).__init__()
        self.config = config
        self.device = torch.device("cuda:{}".format(config.gpu))
        self.embedding = nn.Embedding.from_pretrained(pretrained_weights)
        self.embedding.weight.requires_grad = False

        self.out_channel = config.out_channel
        self.conv3 = nn.Conv2d(1, 32, (3, config.word_embedding_dimension))
        self.conv4 = nn.Conv2d(1, 32, (4, config.word_embedding_dimension))
        self.conv5 = nn.Conv2d(1, 32, (5, config.word_embedding_dimension))
        self.conv6 = nn.Conv2d(1, 32, (6, config.word_embedding_dimension))
        self.Max3_pool = nn.MaxPool2d((self.config.sentence_max_size-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((self.config.sentence_max_size-4+1, 1))
        self.Max5_pool = nn.MaxPool2d((self.config.sentence_max_size-5+1, 1))
        self.Max6_pool = nn.MaxPool2d((self.config.sentence_max_size-6+1, 1))

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, config.label_num)

    def forward(self, x, x_len):
        batch = len(x)
        # self.device = torch.device("cuda:{}".format(0))
        tmp_list = [0 for i in range(self.config.sentence_max_size - len(x[0]))]
        x[0].extend(tmp_list)

        x = [torch.LongTensor(x[i]).to(self.device) for i in range(batch)]

        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        
        # indice to embedding
        x = self.embedding(x)
        x = x.unsqueeze(1)
        # print(x.size())
        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))
        x4 = F.relu(self.conv6(x))

        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)
        x4 = self.Max6_pool(x4)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3, x4), -1)
        # print(x.size())
        x = x.view(batch, 1, -1)
        # print(x.size())

        # project the features to the labels
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        x = x.view(-1, self.config.label_num)

        return x


if __name__ == '__main__':
    print('running the TextCNN...')