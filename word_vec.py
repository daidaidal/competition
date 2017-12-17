# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging

#1.The word embedding vector of wi
#2.The real-valued embedding vector for the entitytype of wi
#3.The binary vector whose dimensions correspond to the possible relations between words in the dependency trees

input = "She was so happy that she ran to the forest like a deerlet."

#1.自己训练词向量
 
 
# 主程序
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
sentences =word2vec.Text8Corpus(u"C:/trial_data/input/word.txt")  # 加载语料
model =word2vec.Word2Vec(sentences, size=200)  #训练skip-gram模型，默认window=5
model.save(u"C:/trial_data/input/word_vec.model")



