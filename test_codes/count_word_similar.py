import os
import sys
import time
import re
docu = open('../word_similat2.txt', 'r', encoding='UTF-8')
dic_injure = {}
dic_attack = {}
dic_die = {}

dic_t = {"injure":0,"attack":1,"die":2,"0":3}

for line in docu:
    line_word = line.split()
    tf = line_word[0]
    input_word = line_word[1]
    predic_type = line_word[2]
    true_predic = line_word[3]

    if predic_type == 'injure':
        if input_word in dic_injure:
            dic_injure[input_word][dic_t[true_predic]] = dic_injure[input_word][dic_t[true_predic]] + 1
        else:
            temp_list = [0,0,0,0]
            temp_list[dic_t[true_predic]] = 1
            dic_injure[input_word] = temp_list

    elif predic_type == "attack":
        if input_word in dic_attack:
            dic_attack[input_word][dic_t[true_predic]] = dic_attack[input_word][dic_t[true_predic]] + 1
        else:
            temp_list = [0,0,0,0]
            temp_list[dic_t[true_predic]] = 1
            dic_attack[input_word] = temp_list

    elif predic_type == "die":
        if input_word in dic_die:
            dic_die[input_word][dic_t[true_predic]] = dic_die[input_word][dic_t[true_predic]] + 1
        else:
            temp_list = [0,0,0,0]
            temp_list[dic_t[true_predic]] = 1
            dic_die[input_word] = temp_list

print(dic_injure)
print(dic_attack)
print(dic_die)

docu.close()