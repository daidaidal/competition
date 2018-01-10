import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from gensim.models import word2vec
try:
    daisfs
except Exception:
    print("i'm here")

# a = torch.FloatTensor([1,2,3,4,5])
# print(nn.LogSoftmax(a,1))

# loss = nn.CrossEntropyLoss()
# input1 = Variable(torch.randn(3, 5), requires_grad=True)
# target = Variable(torch.LongTensor(3).random_(5))
# output = loss(input1, target)
# output.backward()
# print('ok')

# model = word2vec.Word2Vec.load(u"/home/sfdai/word_vec.model")
# try:
#     c = model['wdljalkfj']
# except KeyError:
#     c = "a"
# print(c)

# a = np.array([1,2,3,4,5])
# b = np.argmax(a)
# c=torch.LongTensor(3).random_(5)
# d = c.view(-1,3)
# print(c)
# print(d)
# print(torch.from_numpy(a))

# rnn = nn.GRU(10, 20, 2,batch_first=True)
# input1 = Variable(torch.randn(5, 3, 10))
# h0 = Variable(torch.randn(2, 3, 20))
# output, hn = rnn(input1, h0) #5 3 20
# print("ok")