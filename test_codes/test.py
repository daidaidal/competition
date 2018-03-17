import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from gensim.models import word2vec
import torch
import os
from nltk.parse import stanford
import time
import gru
from gensim.models import word2vec
a = 1
def f():
    global a
    print(a)
f()
# model = word2vec.Word2Vec.load(u"/home/sfdai/word2vec_gensim")
# similar_type_injure = ["wound","injure","hurt","disabled","hospitalized"]
# similar_type_die = ["die","killed","assassination","late","fatal","suicide"]
# similar_type_attack = ["bomb","exploded","gunfire","shot","fire","clashes","attack","battle","throw","blow"]
# list_sum = [similar_type_injure,similar_type_attack,similar_type_die]
#
# write_list= []
# for list in list_sum:
#     count = 1
#     list_t = []
#     for word in list:
#         for similar_word in model.most_similar(word):
#             list_t.append(similar_word[0])
#     print(list_t)
#     list_t.clear()

# a = torch.FloatTensor([1,2])
# torch.save(a,'
# print("success")
# b = torch.load('a')
# print(b)
# a = torch.FloatTensor([3,4])
# torch.save(a,'a')
# b = torch.load('a')
# print(b)

# a0 = Variable(torch.FloatTensor([2]),requires_grad=True)
# a = Variable(torch.FloatTensor([1]),requires_grad=True)
# b = a*a+2+a0*a0*a0
# b.backward()
# print(a.grad)
# print(a0.grad)
# c = Variable(torch.FloatTensor([3]),requires_grad=True)
# b = a*a+2+a0*a0*a0+c
# b.backward()
# print(a.grad/2)
# print(a0.grad/2)


# a.ze
# print(a.grad.data)
# print(a0.grad.data)




# dic1 = {'a':1,'b':2}
# for i in dic1:
#     print(i[0])
# # -*- coding: utf-8 -*-
# import torch
# import os
# from nltk.parse import stanford
#
# Hyper Parameters
# DEP_VEC_LEN = 200
# TRIGGER_CANDIDATE_LIST = ['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']
# # 添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
# os.environ['STANFORD_PARSER'] = '/home/sfdai/jars/stanford-parser.jar'
# os.environ['STANFORD_MODELS'] = '/home/sfdai/jars/stanford-parser-3.8.0-models.jar'
#
# # 为JAVAHOME添加环境变量
# java_path = "/usr/lib/jvm/java-8-oracle/jre/bin/java"
# os.environ['JAVAHOME'] = java_path
# dependency_parser=stanford.StanfordDependencyParser( model_path="/home/sfdai/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
#
# def dependence_matrix(input_sentence):
# # 句法标注
# #    stanford.DependencyGraph
# # input_sentence = "Hello , My name is Melroy ." # there is a full stop!
# # dependency_parser=stanford.StanfordDependencyParser( model_path="/home/sfdai/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
#     sentences = dependency_parser.raw_parse(input_sentence)
#     # print(sentences)
#     # print(input_sentence.split())
#     output_list=[]
#     trigger_candidate_list = []
#     for line in sentences:
#         print(line)
#
#
#         # for i in line:
#         #     out = torch.zeros(DEP_VEC_LEN)
#         #     print(line.nodes[i])
#             # dic = line.nodes[i]['deps']
#             # tags = line.nodes[i]['ctag']
#             # if tags in TRIGGER_CANDIDATE_LIST:
#             #     trigger_candidate_list.append(i-1)
#             # for j in dic:
#             #     out[dic[j][0]]=1
#             # output_list.append(out)
#     # output_matrix = torch.stack(output_list,0)  #size (long_of_sentence,DEP_VEC_LEN)
#     # print(output_matrix.size())
#     return 1
# dependence_matrix("hello , how are you ? i'm fine thank you .")


# input_sentence = "hello , how are you ?"
# trigger_word = "are you"
# split_sentence = input_sentence.split()
# trigger_index = []
# for j in range(2):
#     # try:
#     #     trigger_index.append(split_sentence.index(trigger_word.split()[j]))
#     # except ValueError:
#     a = input_sentence.find(trigger_word.split()[j])
#     b = input_sentence.count(' ',0,a)
#     print(a)
#     print(b)
#     trigger_index.append(b)
# print(trigger_index)

# a = torch.FloatTensor([1,2,3,4,6])
# b = torch.max(a,0)
# print(b[1])

# time1 = time.localtime(time.time())
# print(str(time1[3]) + " " + str(time1[4])+'\n')

# a = " ".join(time1)
# wf = open('/home/sfdai/competition/result_data.txt', 'a+', encoding='UTF-8')
# wf.write(a)
# wf.close()

# TRIGGER_CANDIDATE_LIST = ['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']

# try:
#     train(0.01)
# except Exception:
#     print("1")
#     try:
#         train(0.005)
#     except Exception:
#         print("2")

# a = "hello\nhow are you\ni'm fine thanks"
# wf = open('/home/sfdai/competition/word.txt','a+', encoding='UTF-8')
# wf.write(a)
# wf.close()
# print("write success")
# # os.remove('/home/sfdai/competition/word.txt')
# os.rename('/home/sfdai/competition/word.txt','/home/sfdai/competition/new_word.txt')

# # Hyper Parameters
# DEP_VEC_LEN = 200
#
# # 添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
# os.environ['STANFORD_PARSER'] = '/home/sfdai/jars/stanford-parser.jar'
# os.environ['STANFORD_MODELS'] = '/home/sfdai/jars/stanford-parser-3.8.0-models.jar'
#
# # 为JAVAHOME添加环境变量
# java_path = "/usr/lib/jvm/java-8-oracle/jre/bin/java"
# os.environ['JAVAHOME'] = java_path
# dependency_parser=stanford.StanfordDependencyParser( model_path="/home/sfdai/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
#
# def dependence_matrix(input_sentence):
# # 句法标注
# #    stanford.DependencyGraph
# # input_sentence = "Hello , My name is Melroy ." # there is a full stop!
# # dependency_parser=stanford.StanfordDependencyParser( model_path="/home/sfdai/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
#     sentences = dependency_parser.raw_parse(input_sentence)
#     # print(sentences)
#     # print(input_sentence.split())
#     output_list=[]
#     trigger_candidate_list = []
#     for line in sentences:
#         # print(line)
#         for i in range(1,len(input_sentence.split())+1):
#             out = torch.zeros(DEP_VEC_LEN)
#             dic = line.nodes[i]['deps']
#             tags = line.nodes[i]['ctag']
#             if tags in TRIGGER_CANDIDATE_LIST:
#                 trigger_candidate_list.append(i-1)
#             for j in dic:
#                 out[dic[j][0]]=1
#             output_list.append(out)
#     output_matrix = torch.stack(output_list,0)  #size (long_of_sentence,DEP_VEC_LEN)
#     # print(output_matrix.size())
#     return output_matrix,trigger_candidate_list
# a,b = dependence_matrix("A cute cat is eating a small fish .")
# print(a)
# print(b)

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