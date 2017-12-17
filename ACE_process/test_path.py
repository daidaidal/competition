# -*- coding: utf-8 -*-
import os
from xml.etree import ElementTree as ET
import all_in_one

rootdir = 'C:/Users/Daisf/Documents/python_project/competition/ACE 2005 Multilingual Training Data V6.0/ACE2005-TrainingData-V6.0/Englishref/bn/timex2norm'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isfile(path) and path[-3:] == "xml":
        tree = ET.parse(path)
        root = tree.getroot()
        txt = []
        for child in root:  # document
            for child1 in child:  # entity,timex2,relation,event
                if child1.tag == 'event':
                    for child2 in child1:  # event_argument,event_mention
                        if child2.tag == 'event_mention':
                            input_sentence = ''
                            trigger = ''
                            argument_role = {}
                            for child3 in child2:  # extent(i don't know it),ldc_scope(whole words),anchor(trigger),event_mention_argument(argument_role)
                                if child3.tag == 'ldc_scope':
                                    input_sentence = child3[0].text
                                elif child3.tag == 'anchor':
                                    trigger = child3[0].text
                                elif child3.tag == 'event_mention_argument':
                                    argument_role[child3[0][0].text] = child3.attrib['ROLE']
                            all_in_one.training(input_sentence,trigger,argument_role)
