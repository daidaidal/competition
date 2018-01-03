import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
a = np.array([1,2,3,4,5])
b = np.argmax(a)
c=torch.LongTensor(3).random_(5)
d = c.view(-1,3)
print(c)
print(d)
print(torch.from_numpy(a))
