# -*- coding: utf-8 -*-
import torch
import os
from nltk.parse import stanford

# Hyper Parameters
DEP_VEC_LEN = 100

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
            out = torch.zeros(DEP_VEC_LEN)
            dic = line.nodes[i]['deps']
            for j in dic:
                out[dic[j][0]]=1
            output_list.append(out)
    output_matrix = torch.stack(output_list,0)  #size (long_of_sentence,DEP_VEC_LEN)
    # print(output_matrix.size())
    return output_matrix