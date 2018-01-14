# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# Hyper Parameters

class NET(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(NET, self).__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)


    def forward(self, start_input):
        h_0 =self.input_layer(start_input)
        h_1 = F.tanh(h_0)
        h_2 = self.output_layer(h_1)
        return h_2


# f = NET()
# out = []
# loss_func = nn.CrossEntropyLoss()
# for i in range(1):
#     input1 = Variable(torch.rand(1633),requires_grad=True)
#     # target_output = Variable(torch.LongTensor([1]))
#     output1 = f(input1)
#     out.append(output1)
#
# target_output = Variable(torch.LongTensor([1]))
# output2 = torch.stack(out,dim=0)
# loss = loss_func(output2,target_output)
# loss.backward()
# print ("successs!")
