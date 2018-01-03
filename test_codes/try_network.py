import torch
from torch import nn
from torch.autograd import Variable
import dep_paser
import numpy as np

class NET(nn.Module):
    def __init__(self,input_size=5,hidden_size=4,output_size=3):
        super(NET, self).__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.output_layer = torch.nn.Linear(hidden_size, output_size)
        self.lmax = torch.nn.LogSoftmax()
        self.smax = torch.nn.Softmax()

    def forward(self, start_input):
        h_1 = self.output_layer(self.sigmoid(self.input_layer(start_input)))
        h_2 = self.smax(h_1)
        return h_2

n = NET()

input1 = Variable(torch.FloatTensor([1,2,3,4,5]))
c = torch.nn.LogSoftmax()
d = torch.nn.Softmax()
print(input1)
output1 = d(input1)
print(output1)
output2= c(input1)
print(output2)
loss_func = torch.nn.NLLLoss()
loss_func = torch.nn.CrossEntropyLoss()

