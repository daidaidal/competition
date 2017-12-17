from xml.etree import ElementTree as ET

tree = ET.parse('AFP_ENG_20030304.0250.apf.xml')
root = tree.getroot()
txt = []
for child in root: #document
    for child1 in child: #entity,timex2,relation,event
       if child1.tag == 'event':
           for child2 in child1: # event_argument,event_mention
               if child2.tag == 'event_mention':
                   input_sentence = ''
                   trigger = ''
                   argument_role = {}
                   for child3 in child2: # extent(i don't know it),ldc_scope(whole words),anchor(trigger),event_mention_argument(argument_role)
                       if child3.tag == 'ldc_scope':
                           input_sentence = child3[0].text
                           txt.append('input sentence:')
                           txt.append(input_sentence + '\n')
                       elif child3.tag == 'anchor':
                           trigger = child3[0].text
                           txt.append('trigger:')
                           txt.append(trigger + '\n')
                       elif child3.tag == 'event_mention_argument':
                           txt.append(child3[0][0].text+':')
                           txt.append(child3.attrib['ROLE'] + '\n')

wf = open('C:/Users/Daisf/Documents/python_project/competition/ACE_process/test_write.txt','w+', encoding='UTF-8')
w = " ".join(txt)
wf.write(w)
wf.close()
                       
                       
               