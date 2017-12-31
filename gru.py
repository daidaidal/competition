# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import dep_paser
import numpy as np

from gensim.models import word2vec

model = word2vec.Word2Vec.load(u"C:/trial_data/input/word_vec.model")

torch.manual_seed(1)    # reproducible
# input_sentence="There is a book on the shoe cabinet"

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
HIDDEN_SIZE = 512
# TIME_STEP = len(input_sentence.split())         # rnn time step / image height
INPUT_SIZE = 300        # rnn input size / image width
LR = 0.5               # learning rate
OUTPUT_SIZE = 300


def pretreatment(sentence_arg,judge=True):
    sentence_arg = "hello , what is your name ? "
    sentence_arg_list = sentence_arg.split()
    if not judge:
        sentence_arg_list.reverse()

    dep_vec = dep_paser.dependence_matrix(sentence_arg) #sentence_len*100
    sentence_len = len(sentence_arg_list)
    sentence_vec_list = []
    for i in range(0,sentence_len):
        try:
            t = model[sentence_arg_list[i]]
        except Exception:
            t = np.zeros(200)
        sentence_vec_list.append(torch.from_numpy(t))
    word_vec = torch.stack(sentence_vec_list,0) # sentence_len*200
    return torch.cat((dep_vec,word_vec),1) #sentence_len*300


class RNN(nn.Module):
    def __init__(self,num_layers=1):
        super(RNN, self).__init__()

        self.rnn = nn.GRU(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=num_layers,           # number of rnn layer
            batch_first=True,
        )

        self.out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, start_input, hidden_state,judge=True):
        # **output** (seq_len, batch, hidden_size * num_directions)
        r_out,hidden_state = self.rnn(start_input, hidden_state)   # None represents zero initial hidden state
        outs=[]
        for time_step in range(r_out.size(1)):  # 对每一个时间点计算 output
            outs.append(self.out(r_out[:, time_step, :]))
            if not judge:
                outs.reverse()
        return torch.stack(outs, dim=1), hidden_state

