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
os.environ['CLASSPATH'] = 'C:/Users/Daisf/Documents/jars/stanford-ner.jar'
# os.environ['STANFORD_MODELS'] = 'C:/Users/Daisf/Documents/jars/stanford-parser-3.8.0-models.jar'

# 为JAVAHOME添加环境变量
java_path = "C:/Program Files/Java/jdk1.8.0_144/bin/java.exe"
os.environ['JAVAHOME'] = java_path
# english.all.3class.distsim.crf.ser.gz
# english.conll.4class.distsim.crf.ser.gz
# english.muc.7class.distsim.crf.ser.gz
# example.serialized.ncc.ncc.ser.gz

eng_tagger1 = StanfordNERTagger('C:/Users/Daisf/Documents/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz')
eng_tagger2 = StanfordNERTagger('C:/Users/Daisf/Documents/stanford-ner-2017-06-09/classifiers/english.conll.4class.distsim.crf.ser.gz')
eng_tagger3 = StanfordNERTagger('C:/Users/Daisf/Documents/stanford-ner-2017-06-09/classifiers/english.muc.7class.distsim.crf.ser.gz')
eng_tagger4 = StanfordNERTagger('C:/Users/Daisf/Documents/stanford-ner-2017-06-09/classifiers/example.serialized.ncc.ncc.ser.gz')

a = eng_tagger1.tag('Rami Eid is studying at Stony Brook University in NY .'.split())
print(a[0][0])
def get_entity(arg_sentence): # format dic:{ 1:(start,len),2:(start,len)}
    tag = eng_tagger1.tag(arg_sentence.split())
    last='O'
    start_index = -1
    len_entity = 0
    dic = {}
    count = 0
    for i in range(len(arg_sentence.split())):
        if tag[i][1] != 'O':
            if tag[i][1] == last:
                len_entity += 1
                continue
            else:
                if start_index>0 and len_entity>0:
                    count += 1
                    dic[count] = (start_index,len_entity,last)
                start_index = i
                len_entity = 1
                last = tag[i][1]
    return dic,count