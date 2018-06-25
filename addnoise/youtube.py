import json
import re
import os
import sys

path = '/Users/cxs/Downloads/11/'

def mkdir(path):

  exist = os.path.exists(path)
  if not exist:
    os.mkdir(path)
    print('a new folder')
  else:
    print('it is existed')


#File1 = open(r'ytb_url.txt', 'w')
#File2 = open(r'sta_time.txt', 'w')
#File3 = open(r'end_time.txt', 'w')
with open("ontology.json",'r') as load_f:
    load_dict = json.load(load_f)
    print(len(load_dict))
    #print(len(load_dict[610]['positive_examples']))
temp = load_dict[2]['positive_examples']
#for j in range(len(temp)):
#print(pattern2.findall(pattern1.findall(temp[j])[0])[0])

i = 0
for i in range(len(load_dict)):
    if load_dict[i]['positive_examples']:
      mkdir(path + str(load_dict[i]['name']))
      File1 = open(path + str(load_dict[i]['name']) + '/url.txt', 'w')
      j = 0
      for j in range(len(load_dict[i]['positive_examples'])):
        File1.write(load_dict[i]['positive_examples'][j] + '\n')
      File1.close()   