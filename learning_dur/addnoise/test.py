import re
from scipy.io import wavfile
import scipy
import random
import numpy as np
import os
import argparse

pure_signal = '/Users/cxs/Downloads/addnoise/vad/Wechat010116000051-1457752058554.wav'
father_noisedata_path = '/Users/cxs/Downloads/addnoise/1.wav'
vad_text_path = '/Users/cxs/Downloads/addnoise/vad/Wechatvad.txt'
write_path = ''

def read_signal(signal):
    name, extension = os.path.splitext(signal)
    if extension == '.wav':
        fs, data = scipy.io.wavfile.read(signal)
        data = data/max(abs(data))
        return data
    elif extension == '.pcm':
        data = np.memmap(signal,dtype='h', mode = 'r')
        data = np.asarray(data, dtype=np.int16)
        data = data/max(abs(data))
        return data

def write_signal(write_path, data, sample_rate=16000):
        if os.path.exists(write_path):
            scipy.io.wavfile.write(write_path, sample_rate, data)
        else:
            os.mkdir(write_path)
            scipy.io.wavfile.write(write_path, sample_rate, data)
            

def write_signal(write_path, data, sample_rate=16000):
        scipy.io.wavfile.write(write_path, sample_rate, data)

def clip(p_wavedata, n_wavedata):
    n = len(n_wavedata)
    s = len(p_wavedata)
    temp = dict()
    if n < s:
        n_wavedata = np.hstack([n_wavedata,n_wavedata[0, 0:(s-n)]])
        #n_wavedata = n_wavedata.extend(n_wavedata[0:(s-n)])
    else:
        start_point = random.randint(0, n - s)

        n_wavedata = n_wavedata[start_point:start_point + s]
    temp = {'start_index':start_point, 'end_index':start_point + s}

    return n_wavedata, temp

def calc_rou(p_wavedata, n_wavedata, snr):
    n_wavedata = clip(p_wavedata, n_wavedata)
    power_s = np.sum(p_wavedata**2.)/len(p_wavedata)
    power_n = np.sum(n_wavedata**2.)/len(n_wavedata)

    snr_num = 10.**(snr/10.)
    rou = np.sqrt((power_s/power_n)/snr_num)

    return rou

with open(vadclearlistpath, 'r') as f:
    f_lines = f.readlines()
    for i in range(len(f_lines)):
        pure_signal = f_lines[i].split()[0]
        data = read_signal(pure_signal)
        vad_clip=[]
        #print(pure_signal)
        for j in range(1,len(f_lines[i].split())):
                temp = re.findall(r'\d{1,5}', f_lines[i].split()[j])
                vad_clip.append(temp)

        vad_data = []
        for z in range(len(vad_clip)):
                vad = [int(x)*16 for x in vad_clip[z]]
                data = list(data)
                vad_data.extend(data[vad[0]:vad[1]])
        vad_data = np.array(vad_data)

        with open(vadnoiselistpath, 'r') as ff:
                file_name = ff.readlines()

                chose_one_file = random.sample(file_name, 1)

                noise_signal = chose_one_file[0].split()[0]
                noisedata = read_signal(noise_signal)
                noisedata, clip_vad = clip(data, noisedata)
                noisedata = noisedata[:, 0]

                vadnoise_clip=[]
            
                for m in range(1,len(chose_one_file[0].split())):
                        tempnoise = re.findall(r'\d{1,5}', chose_one_file[0].split()[m])
                        tempnoise = [int(x) for x in tempnoise]
                        if clip_vad['start_index'] <= tempnoise[0]:
                            if clip_vad['end_index'] > tempnoise[0]&clip_vad['end_index'] <= tempnoise[1]:
                                tempnoise[1] = clip_vad['end_index']
                            else:
                                None
                        if clip_vad['start_index'] > tempnoise[0]&clip_vad['start_index'] < tempnoise[1]:
                            if clip_vad['end_index'] <= tempnoise[1]:
                                tempnoise[0] = clip_vad['start_index']
                                tempnoise[1] = clip_vad['end_index']
                            elif clip_vad['end_index'] > tempnoise[1]:
                                tempnoise[0] = clip_vad['start_index']
                        vadnoise_clip.append(tempnoise)

                vad_noisedata = []
                for n in range(len(vadnoise_clip)):
                        vadnoise = [int(y)*16 for y in vadnoise_clip[n]]
                        noisedata = list(noisedata)
                        vad_noisedata.extend(noisedata[vadnoise[0]:vadnoise[1]])
                vad_noisedata = np.array(vad_noisedata)

        snr = random.randint(-5,15)
        #snr = 0
        if list(vad_noisedata):
            rou = calc_rou(vad_data, vad_noisedata, snr)
        else:
            rou = calc_rou(vad_data, noisedata, snr)
        mix_data = data + rou * noisedata

        if max(abs(mix_data)) > 30000:
                a = random.uniform(10000 / max(abs(mix_data)), 25000 / max(abs(mix_data)))
                mix_data = data  + rou * noisedata * a

        mix_data = np.asarray(mix_data, dtype=np.int16)
        write_signal(write_path + str(i) + '.wav', mix_data)

parser = argparse.ArgumentParser()
parser.add_argument('--pure_signal', type = str, help="Directory for storing puredata")
parser.add_argument()