from xml.etree import ElementTree as ET
import os
import re

tree = ET.parse('AFP_ENG_20030304.0250.apf.xml')
root = tree.getroot()
txt = []
rootdirl = 'C:/Users/Daisf/Documents/python_project/competition/ACE 2005 Multilingual Training Data V6.0/ACE2005-TrainingData-V6.0/Englishref/bn/timex2norm'
listl = os.listdir(rootdirl) #列出文件夹下所有的目录与文件
count = 0
percent = 0
trigger_judge = 0
txt = []
for i in range(0,len(listl)):
    path = os.path.join(rootdirl,listl[i])
    if os.path.isfile(path) and path[-7:] == "apf.xml":
        # print(path+'\n')
        tree = ET.parse(path)
        root = tree.getroot()
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
                                    input_sentence = child3[0].text.replace("\n",' ').replace(","," , ")+" . "
                                    txt.append(input_sentence)
                                    # print(input_sentence)
# wf = open('C:/Users/Daisf/Documents/python_project/competition/ACE_process/test_write.txt','a+', encoding='UTF-8')
wf = open('C:/trial_data/input/word.txt','a+', encoding='UTF-8')
w = " ".join(txt)
wf.write(w)
wf.close()
                       
                       
               