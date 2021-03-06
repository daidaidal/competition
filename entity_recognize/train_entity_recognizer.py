# -*- coding: utf-8 -*-
import os
from xml.etree import ElementTree as ET
import all_in_one
import sys
#hello git
def train():
    rootdir_list = ['/home/sfdai/data/Englishref/bc/timex2norm','/home/sfdai/data/Englishref/bn/timex2norm',
               '/home/sfdai/data/Englishref/cts/timex2norm','/home/sfdai/data/Englishref/nw/timex2norm',
               '/home/sfdai/data/Englishref/un/timex2norm']
    count=0
    # ,'/home/sfdai/data/Englishref/wl/timex2norm'
    for rootdir in rootdir_list:
        list1 = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        persent = 0
        trigger_judge = 0
        for i in range(0, len(list1)):
            path = os.path.join(rootdir, list1[i])
            if os.path.isfile(path) and path[-3:] == "xml":
                print(path)
                print('--------')
                tree = ET.parse(path)
                root = tree.getroot()
                txt = []
                for child in root:  # document
                    for child1 in child:  # entity,timex2,relation,event
                        if child1.tag == 'event':
                            trigger_subtype = ''
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
                                    # all_in_one.training(input_sentence, trigger, trigger_subtype, argument_role)
                                    count = count+1
    print(count)



def test():
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    rootdir_list = ['/home/sfdai/data/Englishref/wl/timex2norm']
    for rootdir in rootdir_list:
        list1 = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        count = 0
        persent = 0
        trigger_judge = 0
        for i in range(0, len(list1)):
            path = os.path.join(rootdir, list1[i])
            if os.path.isfile(path) and path[-3:] == "xml":
                print(path)
                print('--------')
                tree = ET.parse(path)
                root = tree.getroot()
                txt = []
                for child in root:  # document
                    for child1 in child:  # entity,timex2,relation,event
                        if child1.tag == 'event':
                            trigger_subtype = ''
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
                                    tpl,fpl,tnl,fnl = all_in_one.testing(input_sentence, trigger, trigger_subtype, argument_role)
                                    tp+=tpl
                                    fp+=fpl
                                    tn+=tnl
                                    fn+=fnl
    print("tp:"+tp)
    print("fp:"+fp)
    print("tn:"+tn)
    print("fn:"+fn)


if __name__=="__main__":
    if sys.argv[1] == "train":
        train()
        print("train success")
    elif sys.argv[1] == "test":
        test()
        print("test success")
else:
    train()

