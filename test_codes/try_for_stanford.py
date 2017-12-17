# -*- coding: utf-8 -*-
import nltk
from nltk.corpus import wordnet
# -*- coding: utf-8 -*-
import os
from nltk.parse import stanford

#添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
os.environ['STANFORD_PARSER'] = 'C:/Users/Daisf/Documents/jars/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'C:/Users/Daisf/Documents/jars/stanford-parser-3.8.0-models.jar'

 
#为JAVAHOME添加环境变量
java_path = "C:/Program Files/Java/jdk1.8.0_144/bin/java.exe"
os.environ['JAVAHOME'] = java_path

#句法标注
a = stanford.D
print(a.tree())
parser = stanford.StanfordParser(model_path="C:/Users/Daisf/Documents/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
sentences = parser.raw_parse_sents(("Hello, My name is Melroy.", "What is your name?"))
print(sentences)

# GUI
for line in sentences:
    for sentence in line:
        print(sentence)
        
#show_syn = wn.synsets('shot')
"""
count=0
with open('C:/Users/Daisf/Documents/trial_data/input/s2/CONLL/2-6869.conll', 'r',encoding='UTF-8') as f:
        for line in f:
            if line.startswith('#begin'):
                docid = line.lstrip("#begin document (").rstrip().rstrip("); ")
                doc="" 
                title=""
                doc_arr=[]
                title_arr=[]
            elif line.startswith("#end"):
                title=" ".join(title_arr)
                text=title
                tokens = nltk.word_tokenize(text)
                tags = nltk.pos_tag(tokens)
                ners = nltk.ne_chunk(tags)
                print (ners)
                print('-------------------------------')
                count=count+1
                if count==10:
                    break
            else:      
                token=line.split()[1]
                if line.split()[2] == 'TITLE':
                    title_arr.append(token)
                else:
                    doc_arr.append(token)
"""
                    
'''
text=title
tokens = nltk.word_tokenize(text)
tags = nltk.pos_tag(tokens)
ners = nltk.ne_chunk(tags)
'''
#print ('%s --- %s' % (str(ners),str(label(ners.node)))
#wordnet.synsets('shot')[0].lemma_names()
