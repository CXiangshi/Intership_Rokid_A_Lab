
# coding: utf-8

# In[45]:


import numpy as np
import sys
import collections
import os


# In[50]:


BATCH_SIZE = 64
DATA_PATH = 'processing_poetry.txt'


# In[94]:


class Dataset(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.data, self.target = self.read_data()
        self.start = 0
        self.length = len(self.data)
        
        
    def read_data(self):
        id_list = []
        with open(DATA_PATH, 'r') as f:
            f_lines = f.readlines()
            for line in f_lines:
                id_list.append([int(num) for num in line.strip().split()])
                
        nums_batch = len(id_list)//self.batch_size
        x_data = []
        y_data = []
        for i in range(nums_batch):
            start = i*self.batch_size
            end = start + self.batch_size
            batch = id_list[start:end]
            max_length = max(map(len, batch))
            tmp_x = np.full((self.batch_size, max_length), 0, dtype=np.int32)
            for row in range(self.batch_size):
                tmp_x[row, :len(batch[row])] = batch[row]
            tmp_y = np.copy(tmp_x)
            tmp_y[:,:-1] = tmp_y[:,1:]
            x_data.append(tmp_x)
            y_data.append(tmp_y)
        return x_data, y_data
        
    def next_batch(self):
        start = self.start
        self.start += 1 
        if self.start >= self.length:
            self.start = 0
        return self.data[start], self.target[start]

def word_to_id(word, id_dict):
    if word in id_dict:
        return id_dict[word]
    else:
        return id_dict['<unknown>']
    


# In[80]:


poetry_list = []

with open('poetry.txt', 'r') as f:
    f_lines = f.readlines()
    print('唐诗总数:{}'.format(len(f_lines)))
    for line in f_lines:
        try:
            title, content = line.split(':')
        except:
            continue
        content = content.strip().replace(' ', '')
        if '(' in content or '(' in content or '<' in content or '《' in content or '_' in content or '[' in content or '“' in content:
            continue
        length = len(content)
        if length < 20 or length > 100:
            continue
        poetry_list.append('s' + content + 'e')

print('用于训练的唐诗数:{}'.format(len(poetry_list)))

poetry_list=sorted(poetry_list,key=lambda x:len(x))
words_list = []

for peotry in poetry_list:
    words_list.extend([word for word in peotry])
    
counter = collections.Counter(words_list)
sorted_words = sorted(counter.items(), key=lambda x:x[1], reverse=True)
words_list = ['<unknow>'] + [x[0] for x in sorted_words]
words_list = words_list[:len(words_list)]

#print('词汇表大小:{}'.format(words_list))

if os.path.exists('poetry.vocab'):
    print('poetry.vocab is existed!')
else:
    with open('poetry.vocab', 'w') as f:
        for word in words_list:
            f.write(word + '\n')

word_to_dict = dict(zip(words_list, range(len(words_list))))
id_list = []
for poetry in poetry_list:
    id_list.append([str(word_to_id(word,word_to_dict)) for word in poetry])
    
if os.path.exists('processing_poetry.txt'):
    print('processing_poetry.txt is existed!')
else:
    with open('processing_poetry.txt', 'w') as f:
        for id_l in id_list:
            f.write(' '.join(id_l) + '\n')


# In[101]:


x = Dataset(batch_size=64)
a, b = x.read_data()
#print(a)


# In[87]:


id_list = []
with open('processing_poetry.txt', 'r') as f:
    f_lines = f.readlines()
    for line in f_lines:
        id_list.append([int(num) for num in line.strip().split()])


# In[89]:


#print(len(id_list)//64)

