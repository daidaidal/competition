import torch
import os
import gru
import numpy as np
import feed_forward_network

from gensim.models import word2vec
from torch.autograd import Variable

# Hyper Paremeters
WINDOW_SIZE = 2
VEC_LEN = 200  # word_emedding

dic_trigger_big = {"LIFE":1,"MOVEMENT":2,"TRANSACTION":3,"BUSINESS":4,"CONFLICT":5,"CONTACT":6,"PERSONELL":7,"JUSTICE":8}
dic_trigger_sub = {"BE-BORN":1,"MARRY":2,"DIVORCE":3,"INJURE":4,"DIE":5,"TRANSPORT":6,"TRANSFER-OWNERSHIP":7,"TRANSFER-MONEY":8,"START-ORG":9,
                   "MERGE-ORG":10,"DECLARE-BANKRUPTCY":11,"END-ORG":12,"ATTACK":13,"DEMONSTRATE":14,"MEET":15,"PHONE-WRITE":16,"START-POSITION":17,
                   "END-POSITION":18,"NOMINATE":19,"ELECT":20,"ARREST-JAIL":21,"RELEASE-PAROLE":22,"TRIAL-HEARING":23,"CHARGE-INDICT":24,"SUE":25,
                   "CONVICT":26,"SENTENCE":27,"FINE":28,"EXECUTE":29,"EXTRADITE":30,"ACQUIT":31,"APPEAL":32,"PARDON":33}

model = word2vec.Word2Vec.load(u"/home/sfdai/word_vec.model")

def getl_trg_i(sentence_vec_list,i):
    vec_sum = []
    length = len(sentence_vec_list)
    for j in range(WINDOW_SIZE*2+1):
        index = i-WINDOW_SIZE+j
        if index < 0 :
            vec_sum.append(torch.zeros(VEC_LEN))
            continue
        if  index > length-1:
            vec_sum.append(torch.zeros(VEC_LEN))
            continue
        vec_sum.append(sentence_vec_list[index])
    return torch.cat(vec_sum) # (WINDOW_SIZE*2+1)*VEC_LEN 1000


def training(input_sentence,trigger_word,trigger_subtype,argument_dic):
    #use gru
    if os.path.exists('/home/sfdai/competition/ACE_process/gru.plk'):
        rnn = torch.load('rnn.pkl')
    else:
        rnn = gru.RNN()
    word_vec_list = gru.get_sentence_vec(input_sentence)
    h_state = Variable(torch.rand(1,1,300))
    h_state_r = Variable(torch.rand(1,1,300))
    # 正向
    x = gru.pretreatment(word_vec_list,input_sentence)
    # print(x.size())
    x = x.view(-1, x.size(0), x.size(1))
    b_x = Variable(x,True)
    output, h_state = rnn(b_x, h_state)
    # 反向
    x_r = gru.pretreatment(word_vec_list,input_sentence,False)
    x_r = x_r.view(-1, x_r.size(0), x_r.size(1))
    b_x_r = Variable(x_r,True)
    output_r, h_state_r = rnn(b_x_r, h_state_r,False)
    h_sum = torch.cat((output, output_r), 2)
    h_sum = h_sum.view(h_sum.size(1),-1) # h_sum[i] to get hi

    split_sentence = input_sentence.split()
    sentence_len = len(split_sentence)
    trigger_index = []
    trigger_count = len(trigger_word.split())
    if trigger_count>1:
        for i in range(trigger_count):
            try:
                trigger_index.append(split_sentence.index(trigger_word.split()[i]))
            except ValueError:
                trigger_index.append(input_sentence.count(' ', input_sentence.find(trigger_word.split()[i])))
    else:
        try:
            trigger_index.append(split_sentence.index(trigger_word))
        except ValueError:
            trigger_index.append(input_sentence.count(' ', input_sentence.find(trigger_word)))

    # trigger_target = np.zeros(33)
    # trigger_target[dic_trigger_sub[trigger_subtype.upper()]] = 1
    # trigger_target = torch.from_numpy(trigger_target)
    trigger_target_num = dic_trigger_sub[trigger_subtype.upper()]-1

    #argument role一定要加上！！！！
    # entity_dic, entity_num = entity_recogenizer.get_entity(input_sentence)

    g_trg = torch.zeros(33)  # 33 types of triggers
    g_trg_arg = torch.zeros(36, 33)  # 40 types of argument roles 33 types of triggers
    if os.path.exists('/home/sfdai/competition/ACE_process/net1.plk'):
        f_network = torch.load('net1.pkl')
    else:
        f_network = feed_forward_network.NET(1633, 600, 33)
    #f_network2 = feed_forward_network.NET(3233, 1600, 36)
    # train
    optimizer_rnn = torch.optim.Adam(rnn.parameters(),lr=0.02)
    optimizer1 = torch.optim.Adam(f_network.parameters(), lr=0.02)  # optimize all cnn parameters
    # optimizer2 = torch.optim.Adam(f_network2.parameters(), lr=0.02)  # optimize all cnn parameters
    # loss_func = torch.nn.CrossEntropyLoss()  # the target label is not one-hotted
    loss_func = torch.nn.CrossEntropyLoss()

    input_list = []
    for i in range(sentence_len):
        input_list.clear()
        input_list.append(h_sum[i].data)
        input_list.append(getl_trg_i(word_vec_list, i))
        input_list.append(g_trg)
        input_cat_vec = torch.cat(input_list, 0)
        input_cat_vec = Variable(input_cat_vec, True)

        output1 = f_network(input_cat_vec)  # h2,h_m
        s = torch.nn.Softmax()
        index_predic_trigger = np.argmax(s(output1).data.numpy())

        # if len(np.where(output1.data.numpy() == 1)[0])==1:
        #     index_predic_trigger = np.where(output1.data.numpy() == 1)[0][0]
        #     g_trg[index_predic_trigger] = 1

        # if i == trigger_index:
        #     judge_trigger = True
        # else:
        #     judge_trigger = False
        #
        # for j in range(entity_num):
        #     input1 = h_sum[i]  # 600
        #     input12 = h_sum[entity_dic[j][0]]  # 600
        #     input2 = getl_trg_i(input_sentence, i)  # 1000
        #     input3 = getl_trg_i(input_sentence, entity_dic[j][0])  # 1000
        #     input4 = g_trg_arg[entity_dic[j][0]]  # 33
        #     input_all = torch.cat(input1, input2, input3, input4)  # 600*2+1000*2+33 = 3233
        #     output2 = f_network2(input_all)
        #
        #     loss2 = loss_func(output2, target_ouput2)  # 计算两者的误差
        #     optimizer2.zero_grad()  # 清空上一步的残余更新参数值
        #     loss2.backward()  # 误差反向传播, 计算参数更新值
        #     optimizer2.step()
        if i in trigger_index:
            target_output = Variable(torch.LongTensor([trigger_target_num]))
        else:
            target_output = Variable(torch.LongTensor([0]))
        # loss
        temp_output = []
        temp_output.append(output1)
        output2 = torch.stack(temp_output)
        loss = loss_func(output2, target_output)  # 计算两者的误差
        optimizer1.zero_grad()  # 清空上一步的残余更新参数值
        optimizer_rnn.zero_grad()
        loss.backward()  # 误差反向传播, 计算参数更新值

        optimizer1.step()
        optimizer_rnn.step()

        torch.save(f_network, 'net1.pkl')  # 保存整个网络
        torch.save(rnn, 'rnn.pkl')

def testing(input_sentence,trigger_word,trigger_subtype,argument_dic):
    #use gru
    rnn = torch.load('rnn.pkl')
    word_vec_list = gru.get_sentence_vec(input_sentence)
    h_state = Variable(torch.rand(1,1,300))
    h_state_r = Variable(torch.rand(1,1,300))
    # 正向
    x = gru.pretreatment(word_vec_list,input_sentence)
    # print(x.size())
    x = x.view(-1, x.size(0), x.size(1))
    b_x = Variable(x,True)
    output, h_state = rnn(b_x, h_state)
    # 反向
    x_r = gru.pretreatment(word_vec_list,input_sentence,False)
    x_r = x_r.view(-1, x_r.size(0), x_r.size(1))
    b_x_r = Variable(x_r,True)
    output_r, h_state_r = rnn(b_x_r, h_state_r,False)
    h_sum = torch.cat((output, output_r), 2)
    h_sum = h_sum.view(h_sum.size(1),-1) # h_sum[i] to get hi

    split_sentence = input_sentence.split()
    sentence_len = len(split_sentence)
    trigger_index = []
    trigger_count = len(trigger_word.split())
    if trigger_count>1:
        for i in range(trigger_count):
            try:
                trigger_index.append(split_sentence.index(trigger_word.split()[i]))
            except ValueError:
                trigger_index.append(input_sentence.count(' ', input_sentence.find(trigger_word.split()[i])))
    else:
        try:
            trigger_index.append(split_sentence.index(trigger_word))
        except ValueError:
            trigger_index.append(input_sentence.count(' ', input_sentence.find(trigger_word)))

    # trigger_target = np.zeros(33)
    # trigger_target[dic_trigger_sub[trigger_subtype.upper()]] = 1
    # trigger_target = torch.from_numpy(trigger_target)
    trigger_target_num = dic_trigger_sub[trigger_subtype.upper()]-1

    #argument role一定要加上！！！！
    # entity_dic, entity_num = entity_recogenizer.get_entity(input_sentence)

    g_trg = torch.zeros(33)  # 33 types of triggers
    g_trg_arg = torch.zeros(36, 33)  # 40 types of argument roles 33 types of triggers
    if os.path.exists('/home/sfdai/competition/ACE_process/net1.plk'):
        f_network = torch.load('net1.pkl')
    else:
        f_network = feed_forward_network.NET(1633, 600, 33)
    #f_network2 = feed_forward_network.NET(3233, 1600, 36)
    # train
    optimizer_rnn = torch.optim.Adam(rnn.parameters(),lr=0.02)
    optimizer1 = torch.optim.Adam(f_network.parameters(), lr=0.02)  # optimize all cnn parameters
    # optimizer2 = torch.optim.Adam(f_network2.parameters(), lr=0.02)  # optimize all cnn parameters
    # loss_func = torch.nn.CrossEntropyLoss()  # the target label is not one-hotted

    input_list = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(sentence_len):
        input_list.clear()
        input_list.append(h_sum[i].data)
        input_list.append(getl_trg_i(word_vec_list, i))
        input_list.append(g_trg)
        input_cat_vec = torch.cat(input_list, 0)
        input_cat_vec = Variable(input_cat_vec, True)

        output1 = f_network(input_cat_vec)  # h2,h_m
        s = torch.nn.Softmax()
        index_predic_trigger = np.argmax(s(output1).data.numpy())
        if index_predic_trigger != 0: # 预测分类正确,是trigger
            if i in trigger_index:
                if index_predic_trigger == trigger_target_num:
                    tp = tp + 1
                else:
                    fp = fp + 1
        if index_predic_trigger == 0:  # 预测分类错误,是trigger
            if i in trigger_index:
                if index_predic_trigger == trigger_target_num:
                    tn = tn + 1
                else:
                    fn = fn + 1
    return tp,fp,tn,fn


























