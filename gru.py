# -*- coding: utf-8 -*-
import torch
import entity_recogenizer
from torch import nn
from torch.autograd import Variable
import dep_paser
import numpy as np
import all_in_one

from gensim.models import word2vec

model = word2vec.Word2Vec.load(u"/home/sfdai/word2vec_gensim")


injure_similar = ['stabbing','wounds', 'laceration', 'bandaged', 'lacerations', 'blow', 'bleeding', 'bruised', 'injuries', 'incapacitate', 'injures', 'restrain', 'maim', 'immobilize', 'hurt',
'irritate', 'harm', 'paralyze', 'hurting', 'harmed', 'injure', 'embarrassed', 'upset', 'scared', 'annoyed', 'shaken', 'injures', 'handicapped', 'disabilities', 'impaired', 'disability',
'paraplegic', 'injured', 'homebound', 'quadriplegic', 'homeless', 'wheelchairs', 'hospitalised', 'diagnosed', 'bedridden', 'paralyzed', 'recuperating', 'recuperated', 'convalescing', 'coma', 'sidelined',
'incapacitated']

attack_similar = ['blow','coup','assassinate','assassinated','rape','ambushed','attacked', 'blast', 'detonating', 'bombing', 'detonated', 'detonation', 'detonate', 'detonator', 'explosion', 'detonated', 'ignited', 'exploding', 'blew', 'sank',
'explode', 'collapsed', 'explosion', 'explodes', 'shelling', 'fusillade', 'firing', 'gunshots', 'shellfire', 'fire', 'snipers', 'salvos', 'bombardment', 'shooting', 'shoot', 'shots', 'stabbed',
'fired', 'filmed', 'gunned', 'shoots', 'fires', 'gunfire', 'firing', 'ablaze', 'conflagration', 'burning', 'explosion', 'smoke', 'firefighters', 'shellfire', 'confrontations', 'scuffles',
'confrontation', 'altercations', 'skirmishes', 'firefights', 'rioting', 'conflict', 'clash', 'assault', 'attacks', 'counterattack', 'raid', 'ambush', 'attacking', 'invasion', 'assaults', 'airstrike',
 'battles', 'skirmish', 'siege', 'fighting', 'combats', 'skirmishes', 'invasion', 'battlefield', 'sieges', 'duel', 'throwing', 'throws', 'threw', 'grab', 'hurl', 'pull', 'knock', 'tossing', 'tossed',
'blows', 'blowing', 'blew', 'wound', 'blown', 'break', 'knock', 'crushing', 'shake']

die_similar = ['killed','perish', 'kill', 'dying', 'dies', 'murdered', 'killing', 'rescued', 'slain', 'massacred',
'assassinations', 'assassinating', 'murder', 'overthrow', 'massacre', 'early', 'eighties', 'seventies', 'nineties',
'nineteenth', 'twentieth', 'eighteenth', 'autumn', 'deadly', 'lethal', 'debilitating', 'catastrophic', 'severe', 'horrific', 'nonfatal', 'paralyzing', 'fatality', 'murder', 'seppuku', 'suicides',
'murders', 'suicidal', 'patricide', 'assassination', 'carjacking']

entity_vec_dic = {'LOCATION':torch.rand(50),'PERSON':torch.rand(50),'ORGANIZATION':torch.rand(50),'MONEY':torch.rand(50),'PERCENT':torch.rand(50),'DATE':torch.rand(50),
'TIME':torch.rand(50)}

bio_dic = {'b':1,'i':2,'e':3}
torch.manual_seed(1)    # reproducible

unable_vec_dic = {}
temp_vec_for_entity = torch.rand(50)

# input_sentence="There is a book on the shoe cabinet"

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
HIDDEN_SIZE = 400
# TIME_STEP = len(input_sentence.split())         # rnn time step / image height
INPUT_SIZE = 400     # rnn input size / image width

FINPUT_SIZE = 2300
FHIDDEN_SIZE = 600
FOUTPUT_SIZE = 34

def get_similar_word(input_word):
    if input_word in injure_similar:
        return 1
    elif input_word in attack_similar:
        return 2
    elif input_word in die_similar:
        return 3
    else:
        return 0

def get_sentence_vec(input_sentence):
    sentence_arg_list = input_sentence.split()
    sentence_len = len(sentence_arg_list)
    sentence_vec_list = []
    for i in range(0, sentence_len):
        word_temp = sentence_arg_list[i]
        try:
            t = torch.from_numpy(model[word_temp])
        except KeyError:
            try:
                t = torch.from_numpy(model[word_temp.lower()])
            except KeyError:
                try:
                    t = unable_vec_dic[word_temp]
                    print("unable_vec_dic success")
                except Exception:
                    t = torch.rand(300)
                    unable_vec_dic[word_temp] = t

        sentence_vec_list.append(t)
    return sentence_vec_list


def pretreatment(sentence_vec_list,raw_sentence,word_list_index_dic,word_list):
    entity_dic = entity_recogenizer.get_entity(raw_sentence)
    entity_list = []
    for i in range(len(word_list)):
        if word_list[i] in entity_dic:
            type_sum = entity_dic[word_list[i]].split()
            temp_vec = entity_vec_dic[type_sum[0]]
            temp_vec[49] = bio_dic[type_sum[1]]
        else:
            temp_vec = temp_vec_for_entity
        entity_list.append(temp_vec)
    entity_vec = torch.stack(entity_list,0)
    dep_vec = dep_paser.dependence_matrix(raw_sentence,word_list_index_dic) #sentence_len*100
    word_vec = torch.stack(sentence_vec_list,0) # sentence_len*200
    return torch.cat((word_vec,entity_vec,dep_vec),1)  # sentence_len*300


class RNN(nn.Module):
    def __init__(self,num_layers=2):
        super(RNN, self).__init__()

        self.rnn = nn.GRU(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,         # rnn hidden unit
            num_layers=num_layers,           # number of rnn layer
            bidirectional=True,
        )

        self.input_layer = torch.nn.Linear(FINPUT_SIZE, FHIDDEN_SIZE)
        self.relu1 = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(FHIDDEN_SIZE, FOUTPUT_SIZE)


    def forward(self, start_input, hidden_state,word_vec_list):
        # **output** (seq_len, batch, hidden_size * num_directions)
        r_out,hidden_state = self.rnn(start_input, hidden_state)   # None represents zero initial hidden state
        outs=[]
        # outs.append(self.out(r_out[:,0,:]))

        # !!!!!r_out is equal to ret!!
        leng = r_out.size(0)
        for time_step in range(leng):  # 对每一个时间点计算 output
            temp = r_out[time_step, :, :]
            outs.append(temp)

        h_sum = torch.stack(outs, dim=0)
        h_sum = h_sum.view(h_sum.size(0), -1)

        input_list = []
        output_list = []

        for i in range(leng):
            # if i not in trigger_candidate_list:
            #     continue
            input_list.clear()
            input_list.append(h_sum[i].data)  # 700
            input_list.append(all_in_one.getl_trg_i(word_vec_list, i))  # 1500
            # input_list.append(g_trg) #33
            input_cat_vec = torch.cat(input_list, 0)
            input_cat_vec = Variable(input_cat_vec, requires_grad=True)

            a1 = self.input_layer(input_cat_vec)  # h2,h_m
            a2 = self.relu1(a1)
            output1 = self.output_layer(a2)

            output_list.append(output1)

        ret = torch.stack(output_list)

        return ret, hidden_state

# a = RNN()
# inputl = Variable(torch.rand(1,5,300))
# hiddenl = Variable(torch.rand(1,1,300))
# outputl,hiddenll = a(inputl,hiddenl)
# print("ok")


