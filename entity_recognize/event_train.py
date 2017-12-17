import torch
from entity_recognize import lstm
import feed_forward_network

from gensim.models import word2vec

model = word2vec.Word2Vec.load(u"C:/trial_data/input/word_vec.model")
# Hyper Paremeters
WINDOW_SIZE = 2
VEC_LEN = 200 # word_emedding
LR = 0.03

input_sentence = "what is your name ."
h_sum = lstm.lstm_vec(input_sentence) # h_sum[i] to get hi
sentence_len = len(input_sentence.split())

print(h_sum[1].size())


h_matrix = torch.zeros(33)  # 33 types of triggers

f_network = feed_forward_network.NET()
#train
optimizer = torch.optim.Adam(f_network.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = torch.nn.CrossEntropyLoss()   # the target label is not one-hotted

for t in range(100):
    for i in range(sentence_len):
        output1 = f_network(torch.cat(h_sum[i],h_matrix))
        g_trg = g_trg+output1

        #loss
        loss = loss_func(output1, target_ouput)  # 计算两者的误差
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()



#     # Memory things
# # vec
# g_trg_i = torch.randn(33)
# # matrix
# g_arg_i = []
# g_arg_trg_i = []
# g_arg_trg_i_1 = torch.randn(100)
#
#
# input_num_i = 2
# # vec  r_trg_i = cat(hi,l_trg_i,g_trg_i-1)
# l_trg_i = getl_trg_i(input_sentence,input_num_i)
# r_trg_i = torch.cat(h_sum.data[input_num_i],l_trg_i,g_arg_trg_i_1)






















