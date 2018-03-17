import torch
import os
import gru
import numpy as np
import feed_forward_network
import dep_paser
from gensim.models import word2vec
from torch.autograd import Variable

# Hyper Paremeters
WINDOW_SIZE = 2
VEC_LEN = 300  # word_emedding
COM = 'competition'

dic_trigger_big = {"LIFE":1,"MOVEMENT":2,"TRANSACTION":3,"BUSINESS":4,"CONFLICT":5,"CONTACT":6,"PERSONELL":7,"JUSTICE":8}

dic_trigger_sub = {"BE-BORN":1,"MARRY":2,"DIVORCE":3,"INJURE":4,"DIE":5,"TRANSPORT":6,"TRANSFER-OWNERSHIP":7,"TRANSFER-MONEY":8,"START-ORG":9,
                  "MERGE-ORG":10,"DECLARE-BANKRUPTCY":11,"END-ORG":12,"ATTACK":13,"DEMONSTRATE":14,"MEET":15,"PHONE-WRITE":16,"START-POSITION":17,
                  "END-POSITION":18,"NOMINATE":19,"ELECT":20,"ARREST-JAIL":21,"RELEASE-PAROLE":22,"TRIAL-HEARING":23,"CHARGE-INDICT":24,"SUE":25,
                  "CONVICT":26,"SENTENCE":27,"FINE":28,"EXECUTE":29,"EXTRADITE":30,"ACQUIT":31,"APPEAL":32,"PARDON":33}

h_state = Variable(torch.rand(4, 1, 400), requires_grad=True)

# dic_trigger_sub = {"INJURE":1,"ATTACK":2,"DIE":3} # "NONE:0"
# trigger_type_list = ["INJURE","ATTACK","DIE"]

def getl_trg_i(sentence_vec_list,i):
    vec_sum = []
    length = len(sentence_vec_list)
    for j in range(WINDOW_SIZE*2+1):
        index = i-WINDOW_SIZE+j
        if index < 0:
            vec_sum.append(torch.rand(VEC_LEN))
            continue
        if  index > length-1:
            vec_sum.append(torch.rand(VEC_LEN))
            continue
        vec_sum.append(sentence_vec_list[index])
    return torch.cat(vec_sum) # (WINDOW_SIZE*2+1)*VEC_LEN 1000


def training(input_sentence,trigger_word_list,trigger_subtype_list,argument_dic,LR,count):
    # control input sentence_len
    # if count/4842 < 0.10:
    #     return h_state

    if len(input_sentence.split()) > 50:
        return

    rnn = gru.RNN()

    if os.path.exists('/home/sfdai/'+COM+'/rnn.pkl'):
        rnn.load_state_dict(torch.load('rnn.pkl'))

    # process input sentence
    try:
        word_list,word_list_index_dic = dep_paser.process_sentence(input_sentence)
    except Exception:
        return

    # if len(trigger_candidate_list) == 0:
    #     return
    process_sentence = " ".join(word_list)
    word_vec_list = gru.get_sentence_vec(process_sentence)

    # wf = open('sentence.txt', 'a+', encoding='UTF-8')
    # wf.write(process_sentence+'\n')
    # wf.close()

    # return h_state

    # correct trigger information process
    split_sentence = process_sentence.split()
    sentence_len = len(split_sentence)
    trigger_index = []
    trigger_dic = {}
    for j in range(len(trigger_word_list)):
        # if trigger_subtype_list[j] not in trigger_type_list:
        #     continue

        temp = trigger_word_list[j].split()
        trigger_count = len(temp)
        if trigger_count > 1:
            for i in range(trigger_count):
                try:
                    indexl = split_sentence.index(temp[i])
                except ValueError:
                    indexl = process_sentence.count(' ', 0, process_sentence.find(temp[i]))
                trigger_index.append(indexl)
                trigger_dic[indexl] = trigger_subtype_list[j]
        else:
            try:
                indexl = split_sentence.index(trigger_word_list[j])
            except ValueError:
                indexl = process_sentence.count(' ', 0, process_sentence.find(trigger_word_list[j]))
            trigger_index.append(indexl)
            trigger_dic[indexl] = trigger_subtype_list[j]

        if len(trigger_index)==0:
            return

    # use gru to extract feature h_statein parameters
    # h_state = Variable(torch.rand(2,1,600),requires_grad=True)

    # 正向
    x = gru.pretreatment(word_vec_list,input_sentence,word_list_index_dic,word_list)
    
    x = x.view(x.size(0),-1, x.size(1))
    b_x = Variable(x,requires_grad=True)
    output, h_state_ret = rnn(b_x, h_state,word_vec_list)

    optimizer_rnn = torch.optim.Adam(rnn.parameters(),lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()

    target_list = []
    for i in range(sentence_len):
        # if i not in trigger_candidate_list:
        #     continue
        if i in trigger_index:
            target_output = torch.LongTensor([dic_trigger_sub[trigger_dic[i]]])
        else:
            target_output = torch.LongTensor([0])
        target_list.append(target_output)

    target_p1 = torch.stack(target_list)
    target_p2 = target_p1.view(-1)
    target = Variable(target_p2)
    loss = loss_func(output, target)  # 计算两者的误差

    optimizer_rnn.zero_grad() # 清空上一步的残余更新参数值

    loss.backward()  # 误差反向传播, 计算参数更新值

    optimizer_rnn.step()

    torch.save(rnn.state_dict(), 'rnn.pkl')
    return

def nottrain(input_sentence,trigger_word_list,trigger_subtype_list,argument_dic):
    # control input sentence_len
    if len(input_sentence.split()) > 50:
        return 0,0,0,0,0,0
    #load network
    rnn = gru.RNN()
    rnn.load_state_dict(torch.load('rnn.pkl'))

    # process input sentence
    try:
        word_list,word_list_index_dic = dep_paser.process_sentence(input_sentence)
    except Exception:
        print("exception in test1")
        return 0,0,0,0,0,0
    process_sentence = " ".join(word_list)
    word_vec_list = gru.get_sentence_vec(process_sentence)

    # process correct trigger information
    split_sentence = process_sentence.split()
    sentence_len = len(split_sentence)
    trigger_index = []
    trigger_dic = {}
    for j in range(len(trigger_word_list)):
        # if trigger_subtype_list[j] not in trigger_type_list:
        #     continue
        temp = trigger_word_list[j].split()
        trigger_count = len(temp)
        if trigger_count > 1:
            for i in range(trigger_count):
                try:
                    indexl = split_sentence.index(temp[i])
                except ValueError:
                    indexl = process_sentence.count(' ', 0, process_sentence.find(temp[i]))
                trigger_index.append(indexl)
                trigger_dic[indexl] = trigger_subtype_list[j]
        else:
            try:
                indexl = split_sentence.index(trigger_word_list[j])
            except ValueError:
                indexl = process_sentence.count(' ', 0, process_sentence.find(trigger_word_list[j]))
            trigger_index.append(indexl)
            trigger_dic[indexl] = trigger_subtype_list[j]

    #use gru , h_state in parameter
    # h_state = Variable(torch.rand(3,1,200))

    # 正向
    try:
        x = gru.pretreatment(word_vec_list,input_sentence,word_list_index_dic,word_list)
    except Exception:
        print("exception in test2")
        return 0,0,0,0,0,0
    x = x.view(x.size(0), -1, x.size(1))
    b_x = Variable(x, requires_grad=True)
    output, h_state_ret = rnn(b_x, h_state, word_vec_list)

    input_list = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    f_similar_right = 0
    f_similar_wrong = 0
    real_i = 0
    for i in range(sentence_len):
        # if i not in trigger_candidate_list:
            # if i in trigger_index:
            #     fn = fn + 1
            # else:
            #     tn = tn + 1
            # continue

        # word_i = split_sentence[i]
        # similar_output = gru.get_similar_word(word_i)
        #
        # s1 = ''
        # if similar_output == 1:
        #     s1 = "injure"
        # elif similar_output == 2:
        #     s1 = "attack"
        # elif similar_output == 3:
        #     s1 = "die"
        #
        # if similar_output != 0:
        #     if i in trigger_index:
        #         if similar_output == dic_trigger_sub[trigger_dic[i]]:
        #             f_similar_right = f_similar_right + 1
        #             #
        #             s = "similar_correct: "+word_i+" "+s1+" "+str(dic_trigger_sub[trigger_dic[i]])+"\n"
        #             wf = open('word_similat2.txt', 'a+', encoding='UTF-8')
        #             wf.write(s)
        #             wf.close()
        #             continue
        #         else:
        #             f_similar_wrong = f_similar_wrong + 1
        #             #
        #             s = "similar_false: " +word_i+" "+s1+" "+str(dic_trigger_sub[trigger_dic[i]])+"\n"
        #             wf = open('word_similat2.txt', 'a+', encoding='UTF-8')
        #             wf.write(s)
        #             wf.close()
        #     else:
        #         f_similar_wrong = f_similar_wrong + 1
        #         #
        #         s = "similar_false: " +word_i+" "+s1+" "+"0"+"\n"
        #         wf = open('word_similat2.txt', 'a+', encoding='UTF-8')
        #         wf.write(s)
        #         wf.close()

        s = torch.nn.Softmax()
        index_predic_trigger = torch.max(s(output[real_i]),0)[1]
        real_i = real_i + 1
        if index_predic_trigger.data[0] != 0: # 预测为trigger
            if i in trigger_index:
                if index_predic_trigger.data[0] == dic_trigger_sub[trigger_dic[i]]:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                fp = fp + 1
        if index_predic_trigger.data[0] == 0:  # 预测不是trigger
            if i in trigger_index:
                fn = fn + 1
            else:
                tn = tn + 1
    return tp,fp,tn,fn,f_similar_right,f_similar_wrong

        # s = torch.nn.Softmax()
        # index_predic_trigger = torch.max(s(output1),0)[1].data[0] # type float

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


























