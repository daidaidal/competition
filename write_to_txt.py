# -*- coding: utf-8 -*-

import json
import os

def main(subtask):
    print("Subtask %s" % subtask)
    
    input_dir = "%s/%s" % ('C:/trial_data/input', subtask) # the directory of the input data
    #output_dir = "%s/%s" % ('C:/trial_data/output', subtask) # the directory of the output data
    with open('%s/questions.json' % input_dir, encoding='UTF-8') as f:
        questions = json.load(f)
    txt=[]
    for q in questions:
        fn='%s/CONLL/%s.conll' % (input_dir, q)
        with open(fn, encoding='UTF-8') as f:
            for line in f:
                if line[0]=='#':
                    continue
                txt.append(line.split()[1])
    return txt
                
    
                
if __name__=="__main__":
    #os.remove('C:/trial_data/input/word.txt')
    for subtask in ["s1","s2","s3"]:
        a = main(subtask)
        w = " ".join(a)
        wf = open('C:/trial_data/input/word.txt','a+', encoding='UTF-8')
        wf.write(w)
        wf.close()
        