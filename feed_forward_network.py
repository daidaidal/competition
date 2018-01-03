# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable

# Hyper Parameters
INPUT_SIZE = 600+1000+33 # ouput_gru,l,g_trg
HIDDEN_SIZE = 600
OUTPUT_SIZE = 33

class NET(nn.Module):
    def __init__(self,input_size=INPUT_SIZE,hidden_size=HIDDEN_SIZE,output_size=OUTPUT_SIZE):
        super(NET, self).__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.output_layer = torch.nn.Linear(hidden_size, output_size)
        # self.soft_max = torch.nn.LogSoftmax(dim=33)


    def forward(self, start_input):
        h_1 = self.sigmoid(self.input_layer(start_input))
        h_2 = self.output_layer(h_1)
        return h_2
