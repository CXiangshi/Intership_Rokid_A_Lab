import numpy as np
from scipy.io import wavfile
#import wave
import os
import sys
import scipy
#import webrtcvad
#import collections
import argparse
import re


#def calc_vad_E(p_data, list):



'''
def calc_snr(pvad_data, n_data):
    pvad_data = abs(pvad_data)
    Ps = np.sum((pvad_data - np.mean(pvad_data))**2)
    Pn = np.sum((n_data - np.mean(n_data))**2)
    snr = 10*np.log10(Ps/Pn)
    return snr
'''

def read_signal(signal):
    name, extension = os.path.splitext(signal)
    if extension == '.wav':
        '''
        parameters = dict()
        audio = wave.open(signal, 'r')
        params = audio.getparams()
        nchannels, sampwidth, framerate, nframes, comptype, compname = params
        parameters = {
            'nchannels':nchannels,
            'sampwidth':sampwidth,
            'framerate':framerate,
            'nframes':nframes,
            'comptype':comptype,
            'compname':compname
        }
        data = audio.readframes(nframes)
        data = np.fromstring(data, dtype=np.int16)
        data = data/max(abs(data))
        audio.close()
        '''
        fs, data = scipy.io.wavfile.read(signal)
        data = data/max(abs(data))
        return data
    elif extension == '.pcm':
        data = np.memmap(signal,dtype='h', mode = 'r')
        data = np.asarray(data, dtype=np.int16)
        data = data/max(abs(data))
        return data
    

def write_signal(write_path, data, sample_rate=16000):
    """
    audio = wave.open(path, 'w')
    audio.setnchannels(1)
    audio.setsampwidth(2)
    audio.setframerate(sample_rate)
    audio.writeframes(data)
    audio.close()
    """
    scipy.io.wavfile.write(write_path, sample_rate, data)

"""   
class Frame(object):
    def __init__(self, bytes, timetamp, duration):
        self.bytes = bytes
        self.timetamp = timetamp
        self.duration = duration

def frame_generator(frame_duration, audio, sample_rate):
    n = int(sample_rate * (frame_duration/1000) * 2)
    offset = 0
    timetamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset+n], timetamp, duration)
        timetamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration, padding_duration, vad, frames):
    frames = list(frames)
    num_padding = int(padding_duration / frame_duration)
    ring_butter = collections.deque(maxlen = num_padding)

    voice_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        ring_butter.append((frame, is_speech))
        for f, s in ring_butter:
            if s:
                voice_frames.append(f)
        ring_butter.clear()

    if voice_frames:
        yield b''.join([f.bytes for f in voice_frames])

def comparion(pvad_data, n_data):
    pvad_data = np.fromstring(pvad_data, dtype=np.int16)
    n_data = np.fromstring(n_data, dtype=np.int16)
    pvad_E = np.sum(np.power(pvad_data, 2))/len(pvad_data)
    n_E = np.sum(np.power(n_data, 2))/len(n_data)
"""

def clip(p_wavedata, n_wavedata):
    n = len(p_wavedata)
    s = len(n_wavedata)
    if n < s:
        noise = np.hstack([n_wavedata,n_wavedata[0, 0:(s-n)]])
    else:
        start_point = random.randint(0, n - s)
        n_wavedata = n_wavedata[start_point:start_point + s]

    return p_wavedata, n_wavedata

def add_noise(pure_signal, noise_signal, snr = 10):
    p_wavedata = read_signal(pure_signal)
    fs ,n_wavedata = read_signal(noise_signal)
    p_wavedata, n_wavedata = clip(p_wavedata, n_wavedata)
    power_s = np.sum(p_wavedata**2.)
    power_n = np.sum(n_wavedata**2.)

    snr_num = 10.**(snr/10.)
    rou = np.sqrt((power_s/power_n)/snr_num)
    pan_wavedata = p_wavedata + rou * n_wavedata

    return pan_wavedata

#vad = webrtcvad.Vad()
#data = read_signal('1.wav')
#print(len(data))
#frames = frame_generator(30, data, )
#segments = vad_collector(para['framerate'], 30, 300, vad, frames)

#for n in segments:
    #a.append(n)
#pvad_data = np.fromstring(a[0], dtype=np.int16)

#pdata, ndata = clip(data, n_data)
#f = pvad_data/max(abs(pvad_data))

#speech = read_signal('pure2.pcm')
#fs2,noise = scipy.io.wavfile.read('n1.wav')
#a,b =clip(speech,noise)

#temp = add_noise('pure2.pcm', 'n1.wav', snr=0)
#temp = np.asarray(temp, dtype=np.int16)
#scipy.io.wavfile.write('10s.wav', 16000, temp)
#temp = np.fromstring(pcm_file, dtype=np.int16)

#a = scipy.io.wavfile.read('pure1.pcm')
#print(type(a))

#n_data = np.fromstring(n_data, dtype=np.int16)
#write_signal('3.wav', b, para['framerate'])


#scipy.io.wavfile.write("2.wav", p_params, pan_data)
#rate, data = scipy.io.wavfile.read()

vad_path = ''



with open('/Users/cxs/Downloads/addnoise/vad/Wechatvad.txt', 'r') as f:
    f_lines = f.readlines()
    for i in range(len(f_lines)):
        temp = f_lines[i].split()
        for j in range(1:len(temp))
            
            = re.findall(r'\=\d{1,5}', 