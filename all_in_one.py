import torch
import dep_paser
import gru
import numpy as np
import feed_forward_network
import entity_recogenizer

from gensim.models import word2vec

# Hyper Paremeters
WINDOW_SIZE = 2
VEC_LEN = 200  # word_emedding

dic_trigger_big = {"LIFE":1,"MOVEMENT":2,"TRANSACTION":3,"BUSINESS":4,"CONFLICT":5,"CONTACT":6,"PERSONELL":7,"JUSTICE":8}
dic_trigger_sub = {"BE-BORN":1,"MARRY":2,"DIVORCE":3,"INJURE":4,"DIE":5,"TRANSPORT":6,"TRANSFER-OWNERSHIP":7,"TRANSFER-MONEY":8,"START-ORG":9,
                   "MERGE-ORG":10,"DECLARE-BANKRUPTCY":11,"END-ORG":12,"ATTACK":13,"DEMONSTRATE":14,"MEET":15,"PHONE-WRITE":16,"START-POSITION":17,
                   "END-POSITION":18,"NOMINATE":19,"ELECT":20,"ASSERT-JAIL":21,"RELEASE-PAROLE":22,"TRIAL-HEARING":23,"CHARGE-INDICT":24,"SUE":25,
                   "CONVICT":26,"SENTENCE":27,"FINE":28,"EXECUTE":29,"EXTRADITE":30,"ACQUIT":31,"APPEAL":32,"PARDON":33}

model = word2vec.Word2Vec.load(u"C:/trial_data/input/word_vec.model")

def getl_trg_i(input_sentence_arg,i):
    vec_sum = []
    input_sentence_list = input_sentence_arg.split()
    length = len(input_sentence_list)
    for j in range(WINDOW_SIZE*2+1):
        index = i-WINDOW_SIZE+j
        if index < 0 :
            vec_sum.append(torch.zeros(VEC_LEN))
            continue
        if  index > length-1:
            vec_sum.append(torch.zeros(VEC_LEN))
            continue
        vec_sum.append(torch.from_numpy(model[input_sentence_list[index]]))
    return torch.cat(vec_sum) # (WINDOW_SIZE*2+1)*VEC_LEN 1000

def training(input_sentence,trigger_word,argument_dic)
    h_sum = gru.gru_vec(input_sentence)  # h_sum[i] to get hi
    sentence_len = len(input_sentence.split())

    print(h_sum[1].size())
    entity_dic, entity_num = entity_recogenizer.eng_tagger1(input_sentence)

    g_trg = torch.zeros(33)  # 33 types of triggers
    g_trg_arg = torch.zeros(36, 33)  # 40 types of argument roles 33 types of triggers
    f_network = feed_forward_network.NET(1633, 600, 33)
    f_network2 = feed_forward_network.NET(3233, 1600, 36)
    # train
    optimizer1 = torch.optim.Adam(f_network.parameters(), lr=0.02)  # optimize all cnn parameters
    optimizer2 = torch.optim.Adam(f_network2.parameters(), lr=0.02)  # optimize all cnn parameters
    loss_func = torch.nn.CrossEntropyLoss()  # the target label is not one-hotted

    for t in range(100):
        for i in range(sentence_len):
            output1 = f_network(torch.cat(h_sum[i], getl_trg_i[i], g_trg))
            g_trg = g_trg + output1
            if True:
                for j in range(entity_num):
                    input1 = h_sum[i]  # 600
                    input12 = h_sum[entity_dic[j][0]]  # 600
                    input2 = getl_trg_i(input_sentence, i)  # 1000
                    input3 = getl_trg_i(input_sentence, entity_dic[j][0])  # 1000
                    input4 = g_trg_arg[entity_dic[j][0]]  # 33
                    input_all = torch.cat(input1, input2, input3, input4)  # 600*2+1000*2+33 = 3233
                    output2 = f_network2(input_all)

                    loss2 = loss_func(output2, target_ouput2)  # 计算两者的误差
                    optimizer2.zero_grad()  # 清空上一步的残余更新参数值
                    loss2.backward()  # 误差反向传播, 计算参数更新值
                    optimizer2.step()

            # loss
            loss = loss_func(output1, target_ouput)  # 计算两者的误差
            optimizer1.zero_grad()  # 清空上一步的残余更新参数值
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer1.step()

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






















