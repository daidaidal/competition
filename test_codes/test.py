import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from gensim.models import word2vec
import torch
import os
from nltk.parse import stanford

# Hyper Parameters
DEP_VEC_LEN = 200

# 添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
os.environ['STANFORD_PARSER'] = '/home/sfdai/jars/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/home/sfdai/jars/stanford-parser-3.8.0-models.jar'

# 为JAVAHOME添加环境变量
java_path = "/usr/lib/jvm/java-8-oracle/jre/bin/java"
os.environ['JAVAHOME'] = java_path
dependency_parser=stanford.StanfordDependencyParser( model_path="/home/sfdai/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

def dependence_matrix(input_sentence):
# 句法标注
#    stanford.DependencyGraph
# input_sentence = "Hello , My name is Melroy ." # there is a full stop!
# dependency_parser=stanford.StanfordDependencyParser( model_path="/home/sfdai/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
    sentences = dependency_parser.raw_parse(input_sentence)
    # print(sentences)
    # print(input_sentence.split())
    output_list=[]
    for line in sentences:
        # print(line)
        for i in range(1,len(input_sentence.split())+1):
            dic = line.nodes[i]
            print(dic)

dependence_matrix("A cute cat is eating a small fish .")

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