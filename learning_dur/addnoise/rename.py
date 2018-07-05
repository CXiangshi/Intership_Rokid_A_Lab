import os, sys
import re
import os.path

nums = 1
path = 'youtube_video'
f = os.listdir(path)
print (f[0])
i = 0
for i in range(len(f)):
    newname = 'noise' + str(nums) + str(os.path.splitext(f[i])[1])
    os.rename(f[i], newname)
    nums += 1
    #os.rename(newname, file)



