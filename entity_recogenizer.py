import torch
import numpy as np
import dep_paser
import gru
import nltk
from nltk.parse import stanford
import os
from nltk.tag import StanfordNERTagger

from nltk.chunk import named_entity


# 添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
os.environ['CLASSPATH'] = '/home/sfdai/jars/stanford-ner.jar'
# os.environ['STANFORD_MODELS'] = '/home/sfdai/jars/stanford-parser-3.8.0-models.jar'

# 为JAVAHOME添加环境变量
java_path = "/usr/lib/jvm/java-8-oracle/jre/bin/java"
os.environ['JAVAHOME'] = java_path
# english.all.3class.distsim.crf.ser.gz
# english.conll.4class.distsim.crf.ser.gz
# english.muc.7class.distsim.crf.ser.gz
# example.serialized.ncc.ncc.ser.gz

eng_tagger1 = StanfordNERTagger('/home/sfdai/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz')
eng_tagger2 = StanfordNERTagger('/home/sfdai/stanford-ner-2017-06-09/classifiers/english.conll.4class.distsim.crf.ser.gz')
eng_tagger3 = StanfordNERTagger('/home/sfdai/stanford-ner-2017-06-09/classifiers/english.muc.7class.distsim.crf.ser.gz')
eng_tagger4 = StanfordNERTagger('/home/sfdai/stanford-ner-2017-06-09/classifiers/example.serialized.ncc.ncc.ser.gz')

def get_entity(arg_sentence): # format dic:{ 1:(start,len),2:(start,len)}
    tag = eng_tagger3.tag(arg_sentence.split())
    last='O'
    start_index = -1
    len_entity = 0
    dic = {}
    count = 0
    i_before = 0
    before_tag = 'O'
    for i in range(len(arg_sentence.split())):
        if tag[i][1] != 'O':
            if i == i_before+1 and tag[i][1] == before_tag: # in
                if i == len(arg_sentence.split()):
                    dic[tag[i][0]] = tag[i][1] + ' e'
                else:
                    dic[tag[i][0]] = tag[i][1]+' i'
            elif i_before !=0: # end and begin
                dic[tag[i][0]] = tag[i][1]+' b'
                dic[tag[i_before][0]] = tag[i_before][1]+' e'
            else:
                dic[tag[i][0]] = tag[i][1]+' b'# first begin

            i_before = i
            before_tag = tag[i][1]
    return dic

# c = get_entity('Rami Eid is studying at Stony Brook University in NY .')
# c = get_entity('a man died when a tank fired in Baghdad.')
# print(c)