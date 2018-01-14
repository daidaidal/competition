# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import dep_paser
import numpy as np

from gensim.models import word2vec

model = word2vec.Word2Vec.load(u"/home/sfdai/word_vec.model")

torch.manual_seed(1)    # reproducible
# input_sentence="There is a book on the shoe cabinet"

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
HIDDEN_SIZE = 600
# TIME_STEP = len(input_sentence.split())         # rnn time step / image height
INPUT_SIZE = 300      # rnn input size / image width

def get_sentence_vec(input_sentence):
    sentence_arg_list = input_sentence.split()
    sentence_len = len(sentence_arg_list)
    sentence_vec_list = []
    for i in range(0, sentence_len):
        try:
            t = torch.from_numpy(model[sentence_arg_list[i]])
        except KeyError:
            t = torch.rand(200)
        sentence_vec_list.append(t)
    return sentence_vec_list


def pretreatment(sentence_vec_list,sentence_arg,judge=True):
    dep_vec = dep_paser.dependence_matrix(sentence_arg) #sentence_len*100
    if not judge:
        sentence_vec_list.reverse()
    word_vec = torch.stack(sentence_vec_list,0) # sentence_len*200
    return torch.cat((dep_vec,word_vec),1)  # sentence_len*300


class RNN(nn.Module):
    def __init__(self,num_layers=2):
        super(RNN, self).__init__()

        self.rnn = nn.GRU(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=num_layers,           # number of rnn layer
            batch_first=True,
        )


    def forward(self, start_input, hidden_state,judge=True):
        # **output** (seq_len, batch, hidden_size * num_directions)
        r_out,hidden_state = self.rnn(start_input, hidden_state)   # None represents zero initial hidden state
        outs=[]
        # outs.append(self.out(r_out[:,0,:]))

        # !!!!!r_out is equal to ret!!
        for time_step in range(r_out.size(1)):  # 对每一个时间点计算 output
            temp = r_out[:, time_step, :]
            outs.append(temp)
        if not judge:
            outs.reverse()
        ret = torch.stack(outs, dim=1)
        return ret, hidden_state


# a = RNN()
# inputl = Variable(torch.rand(1,5,300))
# hiddenl = Variable(torch.rand(1,1,300))
# outputl,hiddenll = a(inputl,hiddenl)
# print("ok")

