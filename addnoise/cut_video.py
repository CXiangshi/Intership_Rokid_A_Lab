from moviepy.editor import *
import os
import json
import re


#path = '/home/xschang/work/ytb_noise/'
path = '/Users/cxs/Downloads/11/'

p1 = r'\=\d{2,3}'
p2 = r'\d{2,3}'
pattern1 = re.compile(p1)
pattern2 = re.compile(p2)

a = 'gGEMCZmcJo0?start=20&end=140.wav'
temp = re.findall(r'\=\d{1,3}', a)
t = re.findall(r'\d{1,3}', str(temp))
print(int(t[0]))


    #print(filename)

    #for f in noisename:
       # i = 1
        #t = pattern2.findall(pattern1.findall(f))
        #audio = AudioFileClip(f).subclip(t,t+10)
        #result = CompositeAudioClip([audio,])
        #result.write_audiofile(str(i) + '.wav', fps=16000)
        #print(f)
print('finished')
    

