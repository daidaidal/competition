
# -*- coding: utf-8 -*-
import os
from xml.etree import ElementTree as ET
import all_in_one
import sys
import time
import re

def write_xml(tree, out_path):
    #    将xml文件写出
    #    tree: xml树
    #    out_path: 写出路径
    tree.write(out_path, encoding="utf-8",xml_declaration=True)

rootdir_list = ['/home/sfdai/data/Englishref/bc/timex2norm', '/home/sfdai/data/Englishref/bn/timex2norm',
                    '/home/sfdai/data/Englishref/cts/timex2norm', '/home/sfdai/data/Englishref/nw/timex2norm',
                    '/home/sfdai/data/Englishref/un/timex2norm','/home/sfdai/data/Englishref/wl/timex2norm']

for rootdir in rootdir_list:
    list1 = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list1)):
        path = os.path.join(rootdir, list1[i])
        if os.path.isfile(path) and path[-7:] == "apf.xml":
            print(path)
            tree = ET.parse(path)
            output_path = path.replace('data','data_speed')
            print(output_path)
            root = tree.getroot()
            child1_delate_list = []
            child2_delate_list = []
            child3_delate_list = []

            for child in root:  # document
                child1_delate_list.clear()
                for child1 in child:  # entity,timex2,relation,event
                    if child1.tag == 'event':
                        trigger_subtype = child1.attrib['SUBTYPE'].upper()
                        child2_delate_list.clear()
                        for child2 in child1:  # event_argument,event_mention
                            if child2.tag == 'event_mention':
                                input_sentence = ''
                                trigger = ''
                                argument_role = {}
                                child3_delate_list.clear()
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
                                    else:
                                        child3_delate_list.append(child3)
                                for c3 in child3_delate_list:
                                    child2.remove(c3)
                            else:
                                child2_delate_list.append(child2)
                        for c2 in child2_delate_list:
                            child1.remove(c2)
                    else:
                        child1_delate_list.append(child1)
                for c1 in child1_delate_list:
                    child.remove(c1)


            write_xml(tree,output_path)