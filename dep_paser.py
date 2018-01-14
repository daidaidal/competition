# -*- coding: utf-8 -*-
import torch
import os
from nltk.parse import stanford

# Hyper Parameters
DEP_VEC_LEN = 100
TRIGGER_CANDIDATE_LIST = ['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']
# 添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
os.environ['STANFORD_PARSER'] = '/home/sfdai/jars/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/home/sfdai/jars/stanford-parser-3.8.0-models.jar'

# 为JAVAHOME添加环境变量
java_path = "/usr/lib/jvm/java-8-oracle/jre/bin/java"
os.environ['JAVAHOME'] = java_path
dependency_parser=stanford.StanfordDependencyParser(model_path="/home/sfdai/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

def process_sentence(raw_sentence):
    sentences = dependency_parser.raw_parse(raw_sentence)
    trigger_candidate_list = []
    word_list = []
    for line in sentences:
        for i in range(1, len(raw_sentence.split()) + 1):
            tags = line.nodes[i]['ctag']
            if tags in TRIGGER_CANDIDATE_LIST:
                trigger_candidate_list.append(i - 1)
            if str(line.nodes[i]['word']) == 'None':
                word_list.append('diia')
            else:
                word_list.append(str(line.nodes[i]['word']))
    if word_list[-1] == 'diia':
        del word_list[-1]
    return trigger_candidate_list, word_list

def dependence_matrix(input_sentence):
    sentences = dependency_parser.raw_parse(input_sentence)
    output_list=[]
    for line in sentences:
        for i in range(1,len(input_sentence.split())+1):
            out = torch.zeros(DEP_VEC_LEN)
            dic = line.nodes[i]['deps']
            for j in dic:
                out[dic[j][0]]=1
            output_list.append(out)
    output_matrix = torch.stack(output_list,0)  #size (long_of_sentence,DEP_VEC_LEN)
    return output_matrix

