# -*- coding: utf-8 -*-
import os
from xml.etree import ElementTree as ET
import all_in_one
import sys
import time
import re

rootdir_list = ['/home/sfdai/data_speed/Englishref/bc/timex2norm',
                '/home/sfdai/data_speed/Englishref/bn/timex2norm',
                '/home/sfdai/data_speed/Englishref/cts/timex2norm',
                '/home/sfdai/data_speed/Englishref/nw/timex2norm',
                '/home/sfdai/data_speed/Englishref/un/timex2norm',
                '/home/sfdai/data_speed/Englishref/wl/timex2norm']

trigger_type_list = ["INJURE","ATTACK","DIE"]

count = 0
before_sentence = ''
trigger_list = []
trigger_subtype_list = []
# ,'/home/sfdai/data_speed/Englishref/wl/timex2norm'
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
                        trigger_subtype = child1.attrib['SUBTYPE'].upper()
                        if trigger_subtype in trigger_type_list:
                            count = count + 1
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
                                    continue
                                before_sentence = input_sentence
                                trigger_list.clear()
                                trigger_list.append(trigger)
                                trigger_subtype_list.clear()
                                trigger_subtype_list.append(trigger_subtype)

print(count)

