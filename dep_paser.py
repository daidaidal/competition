# -*- coding: utf-8 -*-
import torch
import os
from nltk.parse import stanford

# Hyper Parameters
DEP_VEC_LEN = 50
TRIGGER_CANDIDATE_LIST = ['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']
# 添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
os.environ['STANFORD_PARSER'] = '/home/sfdai/jars/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/home/sfdai/jars/stanford-parser-3.8.0-models.jar'

# 为JAVAHOME添加环境变量
java_path = "/usr/lib/jvm/java-8-oracle/jre/bin/java"
os.environ['JAVAHOME'] = java_path
dependency_parser=stanford.StanfordDependencyParser(model_path="/home/sfdai/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

word_not_vec_dic = {"'s":"0", "n't":"not", "'re":"are", "'ve":"have", "'ll":"will", "'m":"am", "'d":"1"}

def process_sentence(raw_sentence):
    sentences = dependency_parser.raw_parse(raw_sentence)
    trigger_candidate_list = []
    word_list = []
    word_list_index_dic = {} # index in dep tree,it is different from word index :dep_index to word_list_index
    for line in sentences:
        count = len(line.nodes)
        for i in range(1, 50):
            if i == count:
                break
            if line.nodes[i]['address'] is None:
                count = count + 1
                continue
            address = line.nodes[i]['address']
            tags = line.nodes[i]['ctag']
            rel = line.nodes[i]['rel']
            if tags in TRIGGER_CANDIDATE_LIST:
                trigger_candidate_list.append(len(word_list))
            if str(line.nodes[i]['word']) == 'None':
                word_list.append('diia')
            else:
                word = str(line.nodes[i]['word'])
                word_list_index_dic[address] = len(word_list)
                if word in word_not_vec_dic:
                    if word_not_vec_dic[word] == "0":
                        if rel == "auxpass":
                            word_list.append("has")
                        elif rel == "aux":
                            word_list.append("is")
                        else:
                            word_list.append(str(line.nodes[i]['word']))
                    elif word_not_vec_dic[word] == "1":
                        if tags == "MD":
                            word_list.append("would")
                        elif tags == "VBD":
                            word_list.append("had")
                        else:
                            word_list.append(str(line.nodes[i]['word']))
                    else:
                        word_list.append(word_not_vec_dic[str(line.nodes[i]['word'])])

                else:
                    word_list.append(str(line.nodes[i]['word']))
    if word_list[-1] == 'diia':
        exit(2)
    return  word_list,word_list_index_dic

def dependence_matrix(raw_sentence,word_list_index_dic):
    sentences = dependency_parser.raw_parse(raw_sentence)
    output_list=[]
    for line in sentences:
        count = len(line.nodes)
        for i in range(1,50):
            if i == count:
                break
            if line.nodes[i]['address'] is None:
                count = count + 1
                continue
            address = line.nodes[i]['address']
            out = torch.zeros(DEP_VEC_LEN)
            dic = line.nodes[i]['deps']
            for j in dic:
                try:
                    out[word_list_index_dic[dic[j][0]]]=1
                except Exception:
                    print("exception in out[word_list_index]")
            out[word_list_index_dic[address]] = 1

            output_list.append(out)
    output_matrix = torch.stack(output_list,0)  #size (long_of_sentence,DEP_VEC_LEN)
    return output_matrix

