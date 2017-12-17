import torch
import numpy as np
import dep_paser
import gru
import nltk
from nltk.parse import stanford
import os
from nltk.tag import StanfordNERTagger

from nltk.chunk import named_entity

# a = named_entity.NEChunkParserTagger()
# b = a.tag('A police officer was killed in New Jersey today.'.split())
# print (b)


# tokens = nltk.word_tokenize('A man was killed in New Jersey today.')
# tokens = nltk.pos_tag(tokens)
# tree = nltk.ne_chunk(tokens)
# for a in tree:
#     print(a.label())



# # 添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
# os.environ['STANFORD_PARSER'] = 'C:/Users/Daisf/Documents/jars/stanford-parser.jar'
# os.environ['STANFORD_MODELS'] = 'C:/Users/Daisf/Documents/jars/stanford-parser-3.8.0-models.jar'
#
# # 为JAVAHOME添加环境变量
# java_path = "C:/Program Files/Java/jdk1.8.0_144/bin/java.exe"
# os.environ['JAVAHOME'] = java_path
# dependency_parser=stanford.StanfordDependencyParser( model_path="C:/Users/Daisf/Documents/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
# a = dependency_parser.raw_parse('A police officer was killed in New Jersey today .')
# for b in a:
#     print(b)


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

print (eng_tagger1.tag('Rami Eid is studying at Stony Brook University in NY.'.split()))
print (eng_tagger2.tag('Rami Eid is studying at Stony Brook University in NY.'.split()))
print (eng_tagger3.tag('Rami Eid is studying at Stony Brook University in NY.'.split()))


# a = np.array([[1,2],[3,4]])
# b = np.array([[5,6],[7,8]])
# c = torch.from_numpy(a)
# d = torch.from_numpy(b)
# e = torch.cat((c,d),0)
# f = torch.cat((c,d),1)
# print(e)
# print(f)
# print(e.size())
# print(f.size())

# a = gru.gru_vec("Hell , I am Jack Ma .")

# tokens = nltk.word_tokenize('A police officer was killed in New Jersey today.')
# tokens = nltk.pos_tag(tokens)
# tree = nltk.ne_chunk(tokens)
# print(tree)