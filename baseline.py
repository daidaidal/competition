# -*- coding: utf-8 -*-
import json
from nltk.corpus import wordnet as wn

wn_synsets={'killing': {'killing.n.02', 'kill.v.01'}, 'injuring': {'injure.v.01', 'injured.a.01'}, 'fire_burning': {'fire.n.01'}, 'job_firing': {'displace.v.03'}} # the set of synsets that correspond to the question's event type, according to the task description

mention_eval_qs = ['1-89170', '2-7074', '3-59191'] # the set of questions annotated with mentions

# Extract the values given for the event properties (time, location, participant) in the question, as keywords
def extract_keywords(qdata):
    keywords=set()
    for a in ["location", "time", "participant"]:
        if a in qdata:
            raw_value=list(qdata[a].values())[0]
            if a=="location": # locations are preprocessed to remove the root of the URI
                keywords.add(raw_value.split('/')[-1].replace('_', ' '))
            else:
                keywords.add(raw_value)
    return keywords

# Obtain the event lemmas based on the event type in the question. This is done by first mapping the event type to its synsets in WordNet, and then extracting the lemmas of those WordNet synsets
def get_event_lemmas(event_type):
    syns = set(wn_synsets[event_type])
    syns = [wn.synset(i) for i in syns]
    lemmas=set()
    for s in syns:
        lemmas |=set([l.replace('_', ' ') for l in s.lemma_names()])
    return lemmas

# This is the main function for processing of a question
# For each document, we create its version as answer document and as noise document. When we get to its end (#end document), then we decide whether to append the noise or the answer document. We choose the answer document if all event property keywords are found and at least one f the event lemmas was found in the document text.
# For the subtask S1, we make sure that the document is one (if it is not, we don't answer the question). Here we assume that all mentions and their documents are correferential (because we know that the total number of answer incidents is 1).
# For the subtask S2, we assume that the answer mentions belonging to the same document are correferential, but the ones belonging to different documents are not.
def process_question(qdata, infile, answer_is_one=False):
    keywords=extract_keywords(qdata)
    event_type=qdata["event_type"]
    event_lemmas=get_event_lemmas(event_type)
    output=""
    answer_docs=set()
    with open(infile, 'r',encoding='UTF-8') as f:
        chain=1
        for line in f:
            if line.startswith('#begin'): # #begin document (98e47689e81907379a827e2782252ce5);
                docid = line.lstrip("#begin document (").rstrip().rstrip("); ")
                doc="" 
                doc_arr=[]
                answer_doc=line
                noise_doc=line # noise_doc包括所有line，如果非问题答案，output取noise_doc，如果是取answer_doc
            elif line.startswith("#end"): # #end document
                answer_doc+=line
                noise_doc+=line
                doc=" ".join(doc_arr)
                doc=doc.replace("NEWLINE", "\n") # 98e47689e81907379a827e2782252ce5.b17.9	NEWLINE	BODY	-
                if all(k in doc for k in keywords) and any(t in doc_arr for t in event_lemmas):
                    output+=answer_doc
                    answer_docs.add(docid)
                    #此处添加事件代码 doc
                    if not answer_is_one: chain+=1 #!! 1.计算事件标志属性 2.比较是否在记录库中 3.如果不在，加入进去
                else:
                    output+=noise_doc
            else:      
                token=line.split()[1]
                doc_arr.append(token)
                noise_doc+=line
                token=line.split()[1] # 98e47689e81907379a827e2782252ce5.t1.1	shot	TITLE	-
                if token in event_lemmas:
                    new_line=line.split()
                    new_line[3]=new_line[3].replace('-','(%d)\n' % chain) #只要是关键词就标注
                    answer_doc+='\t'.join(new_line)
                else:
                    answer_doc+=line

    if answer_is_one:
        if len(answer_docs):
            return output, answer_docs, 1
        else:
            return output, answer_docs, 0
    else: # s2 or s3
        return output, answer_docs, chain-1

def main(subtask):
    print("Subtask %s" % subtask)
        #cd C:\Users\Daisf\.spyder-py3
        #python temp.py C:\Users\Daisf\Documents\trial_data\input C:\Users\Daisf\Documents\trial_data\input
    input_dir = "%s/%s" % ('C:/Users/Daisf/Documents/trial_data/input', subtask) # the directory of the input data
    output_dir = "%s/%s" % ('C:/Users/Daisf/Documents/trial_data/output', subtask) # the directory of the output data
    with open('%s/questions.json' % input_dir, 'r',encoding='UTF-8') as f:
        questions = json.load(f)

    answers_json={}
    for q in questions:
        fn='%s/CONLL/%s.conll' % (input_dir, q)
        conll_output, answer_docs, num_answer=process_question(questions[q], fn, subtask=="s1")
        if subtask=="s1" and num_answer!=1: continue # don't answer if the answer for s1 is incorrect
        if q in mention_eval_qs:
            with open('%s/CONLL/%s.conll' % (output_dir, q), 'w',encoding='UTF-8') as outfile:
                outfile.write(conll_output)
        answers_json[q]={"numerical_answer": num_answer, "answer_docs": list(answer_docs)}
    with open('%s/answers.json' % output_dir, 'w',encoding='UTF-8') as outjson:
        json.dump(answers_json, outjson, indent=4)

if __name__=="__main__":
    for subtask in ["s1", "s2"]:
        main(subtask)
