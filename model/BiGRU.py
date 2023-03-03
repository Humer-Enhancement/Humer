# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F


class BiGRU(nn.Module):

    def __init__(self, config, pretrained_weights):
        super(BiGRU, self).__init__()
        self.config = config
        self.device = torch.device("cuda:{}".format(config.gpu))
        self.embedding = nn.Embedding.from_pretrained(pretrained_weights)
        self.embedding.weight.requires_grad = False
        self.HIDDEN_CELLS = 128
        self.gru = nn.GRU(input_size=200, hidden_size=self.HIDDEN_CELLS, num_layers=2,
                          batch_first=True, dropout=0.1, bidirectional=True)

        self.linear1 = nn.Linear(2 * self.HIDDEN_CELLS, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, config.label_num)

    def forward_once(self, x, input_lengths):
        '''
        x is of the shape (batch_dim, sequence)
        e.g. x = [[i1, i2, i3], [j1, j2, j3, j4]]

        input_lengths is the list that contains the sequence lengths for each sequence
        e.g. input_lengths = [3, 4]
        '''
        sorted_indices = np.flipud(np.argsort(input_lengths))
        input_lengths = np.flipud(np.sort(input_lengths))
        input_lengths = input_lengths.copy()

        ordered_questions = [torch.LongTensor(x[i]).to(self.device) for i in sorted_indices]
        ordered_questions = torch.nn.utils.rnn.pad_sequence(ordered_questions, batch_first=True)

        embeddings = self.embedding(ordered_questions).to(self.device)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, input_lengths, batch_first=True).to(self.device)
        out, hn = self.gru(packed)

        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(
            out, 
            batch_first=True,
            total_length=int(input_lengths[0]))

        result = torch.FloatTensor(unpacked.size()).to(self.device)
        for i, encoded_matrix in enumerate(unpacked):
            result[sorted_indices[i]] = encoded_matrix


        return result

    def forward(self, x, x_length):

        output = self.forward_once(x, x_length.copy()).to(self.device)
        # print(output.size())
        # print(x_length)

        res = torch.zeros(output.size()[0], 2 * self.HIDDEN_CELLS).to(self.device)
        # print(res.size())

        for index in range(output.size()[0]): # every sequence embedding in batch
            res[index] = output[index, x_length[index] - 1, :].to(self.device)
 

        # project the features to the labels
        res = F.relu(self.linear1(res))
        res = F.relu(self.linear2(res))
        res = F.relu(self.linear3(res))
        res = self.linear4(res)
        res = res.view(-1, self.config.label_num)

        return res
