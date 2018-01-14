# -*- coding: utf-8 -*-
import os
from xml.etree import ElementTree as ET
import all_in_one
import sys
import time
import re
import torch
from torch.autograd import Variable

hidden = None
hidden_reverse = None
def train(J,LR=0.01):
    global hidden
    global hidden_reverse
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    if J:
        rootdir_list = ['/home/sfdai/data/Englishref/bc/timex2norm','/home/sfdai/data/Englishref/bn/timex2norm',
                   '/home/sfdai/data/Englishref/cts/timex2norm','/home/sfdai/data/Englishref/nw/timex2norm',
                   '/home/sfdai/data/Englishref/un/timex2norm']
    else:
        rootdir_list = ['/home/sfdai/data/Englishref/wl/timex2norm']
    count = 1
    before_sentence = ''
    trigger_list = []
    trigger_subtype_list = []
    # ,'/home/sfdai/data/Englishref/wl/timex2norm'
    for rootdir in rootdir_list:
        list1 = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list1)):
            path = os.path.join(rootdir, list1[i])
            if os.path.isfile(path) and path[-7:] == "apf.xml":
                print(path)
                tree = ET.parse(path)
                root = tree.getroot()
                for child in root:  # document
                    for child1 in child:  # entity,timex2,relation,event
                        if child1.tag == 'event':
                            trigger_subtype = child1.attrib['SUBTYPE']
                            for child2 in child1:  # event_argument,event_mention
                                if child2.tag == 'event_mention':
                                    input_sentence = ''
                                    trigger = ''
                                    argument_role = {}
                                    for child3 in child2:  # extent(i don't know it),ldc_scope(whole words),anchor(trigger),event_mention_argument(argument_role)
                                        if child3.tag == 'ldc_scope':
                                            input_sentence = child3[0].text
                                            input_sentence = input_sentence.replace('\n', ' ').replace(',', ' ,')
                                            if input_sentence[-1] != '.':
                                                input_sentence += ' .'
                                        elif child3.tag == 'anchor':
                                            trigger = child3[0].text
                                        elif child3.tag == 'event_mention_argument':
                                            temp = child3[0][0].text.replace('\n', ' ')
                                            argument_role[temp] = child3.attrib['ROLE']
                                    if input_sentence == before_sentence:
                                        trigger_list.append(trigger)
                                        trigger_subtype_list.append(trigger_subtype)
                                        count = count + 1
                                        continue
                                    if before_sentence != '':
                                        if J:
                                            hidden, hidden_reverse = all_in_one.training(before_sentence, trigger_list, trigger_subtype_list, argument_role, LR,hidden,hidden_reverse)
                                            hidden = Variable(hidden.data)
                                            hidden_reverse = Variable(hidden_reverse.data)
                                        else:
                                            tpl, fpl, tnl, fnl,hidden,hidden_reverse = all_in_one.nottrain(before_sentence, trigger_list,trigger_subtype_list,argument_role,hidden,hidden_reverse)
                                            tp += tpl
                                            fp += fpl
                                            tn += tnl
                                            fn += fnl
                                            hidden = Variable(hidden.data)
                                            hidden_reverse = Variable(hidden_reverse.data)
                                    before_sentence = input_sentence
                                    trigger_list.clear()
                                    trigger_list.append(trigger)
                                    trigger_subtype_list.clear()
                                    trigger_subtype_list.append(trigger_subtype)
                                    count = count + 1
                if J:
                    print(count/4842)
                else:
                    print(count / 507)
    if not J:
        tp_str = "tp:" + str(tp) + '\n'
        fp_str = "fp:" + str(fp) + '\n'
        tn_str = "tn:" + str(tn) + '\n'
        fn_str = "fn:" + str(fn) + '\n'

        a = tp_str + fp_str + tn_str + fn_str + '\n'
        wf = open('/home/sfdai/competition/result_data.txt', 'a+', encoding='UTF-8')
        wf.write(a)
        wf.close()

def write_time():
    start_time = time.localtime(time.time())
    s = str(start_time[3]) + " " + str(start_time[4]) + '\n'
    wf = open('/home/sfdai/competition/result_data.txt', 'a+', encoding='UTF-8')
    wf.write(s)
    wf.close()

train(False)

if __name__=="__main__":
    if sys.argv[1] == "train":
        write_time()
        train(True,0.01)
        write_time()
        train(False)



    elif sys.argv[1] == "test":
        write_time()
        train(False)
        write_time()

        # write_time()
        # os.rename('/home/sfdai/competition/rnn0.pkl', '/home/sfdai/competition/rnn.pkl')
        # os.rename('/home/sfdai/competition/net0.pkl', '/home/sfdai/competition/net1.pkl')
        # nottrain()
        #
        # write_time()
        # os.remove('/home/sfdai/competition/rnn.pkl')
        # os.remove('/home/sfdai/competition/net1.pkl')
        # os.rename('/home/sfdai/competition/rnn2.pkl', '/home/sfdai/competition/rnn.pkl')
        # os.rename('/home/sfdai/competition/net2.pkl', '/home/sfdai/competition/net1.pkl')
        # nottrain()
        #
        # write_time()
        # os.remove('/home/sfdai/competition/rnn.pkl')
        # os.remove('/home/sfdai/competition/net1.pkl')
        # os.rename('/home/sfdai/competition/rnn3.pkl', '/home/sfdai/competition/rnn.pkl')
        # os.rename('/home/sfdai/competition/net3.pkl', '/home/sfdai/competition/net1.pkl')
        # nottrain()
        #
        # write_time()
        # os.remove('/home/sfdai/competition/rnn.pkl')
        # os.remove('/home/sfdai/competition/net1.pkl')
        # os.rename('/home/sfdai/competition/rnn4.pkl', '/home/sfdai/competition/rnn.pkl')
        # os.rename('/home/sfdai/competition/net4.pkl', '/home/sfdai/competition/net1.pkl')
        # nottrain()
        # write_time()

