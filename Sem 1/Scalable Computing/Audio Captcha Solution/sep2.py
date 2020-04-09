import tts.sapi
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pylab
import librosa.display
import cv2
from pydub import AudioSegment
from PIL import Image, ImageEnhance
from pydub.silence import split_on_silence
import pydub
import pandas as pd
import glob
import shutil

def split_aud_by_dir(src_dir,out_dir):
    l = os.listdir(src_dir)
    m = os.listdir(out_dir)
    if len(m) != 0:
        m = [i[:8] for i in m]
        l = l[l.index(m[-1]+".mp3"):]
    for j in l:
        try:
            sound_file = AudioSegment.from_mp3(src_dir+j)
    #         sound_file = librosa.core.amplitude_to_db(sound_file0)
            audio_chunks = split_on_silence(sound_file, 
                # must be silent for at least half a second
                min_silence_len=25,
                # consider it silent if quieter than -50 dBFS
                silence_thresh=-90
            )

            for i, chunk in enumerate(audio_chunks):
                out_file = out_dir+j.replace(".mp3","")+"_{0}.mp3".format(i)
            #     print "exporting", out_file
                chunk.export(out_file, format="mp3")
        except:
            print("error with "+j)
    return 0

def split_aud(file):
    sound_file = AudioSegment.from_mp3(file)
#         sound_file = librosa.core.amplitude_to_db(sound_file0)
    audio_chunks = split_on_silence(sound_file, 
        # must be silent for at least half a second
        min_silence_len=15,
        # consider it silent if quieter than -50 dBFS
        silence_thresh=-90
    )

    for i, chunk in enumerate(audio_chunks):
        out_file = file.replace(".mp3","")+"_{0}.mp3".format(i)
    #     print "exporting", out_file
        chunk.export(out_file, format="mp3")
    
    return 0

def expand_aud(src_dir):
    l = os.listdir(src_dir)

    c = {}
    for i in l:
        if i[:8] not in c:
            c[i[:8]] = 1
        else:
            c[i[:8]] += 1

    for i in c:
        if c[i] == 6:
            m = [x for x in l if i in x]
            n = []
            for j in m:
                n.append(pydub.utils.mediainfo(src_dir+j)['duration'])
            n = [float(i) for i in n]
            n = sorted(n,reverse=True)
            for j in m:
                if float(pydub.utils.mediainfo(src_dir+j)['duration']) in n[0:2]:
                    split_aud(src_dir+j)
                    os.remove(src_dir+j)
    return 0

def expand_aud2(src_dir):
    l = os.listdir(src_dir)

    c = {}
    for i in l:
        if i[:8] not in c:
            c[i[:8]] = 1
        else:
            c[i[:8]] += 1

    for i in c:
        if c[i] == 7:
            m = [x for x in l if i in x]
            n = []
            for j in m:
                n.append(pydub.utils.mediainfo(src_dir+j)['duration'])
            n = [float(i) for i in n]
            n = sorted(n,reverse=True)
            for j in m:
                if float(pydub.utils.mediainfo(src_dir+j)['duration']) in n[0:1]:
                    split_aud(src_dir+j)
                    os.remove(src_dir+j)
    return 0

def rem_small_aud(src_dir):
    l = os.listdir(src_dir)

    c = {}
    for i in l:
        if i[:8] not in c:
            c[i[:8]] = 1
        else:
            c[i[:8]] += 1

    for i in c:                
        if c[i] > 8:
            m = [x for x in l if i in x]
            n = []
            for j in m:
                n.append(pydub.utils.mediainfo(src_dir+j)['duration'])
            n = [float(i) for i in n]
            n = sorted(n,reverse=True)
#             print(n)
            for j in m:
                if float(pydub.utils.mediainfo(src_dir+j)['duration']) not in n[:8]:
                    os.remove(src_dir+j)
#                     print(src_dir+j)
            
    return 0

def split_aud_final(src_dir, out_dir):
    split_aud_by_dir(src_dir = src_dir, out_dir = out_dir)
    expand_aud(out_dir)
    expand_aud2(out_dir)
    rem_small_aud(out_dir)
    return 0

def func(src_dir,out_dir):
    try:
        return split_aud_final(src_dir = src_dir, out_dir = out_dir)
    except:
        return 'error'

def func2(src_dir,out_dir):
    a = func(src_dir = src_dir, out_dir = out_dir)
    cnt = 0
    while(a == 'error'):
        print('error encountered')
        func(src_dir = src_dir, out_dir = out_dir)
        cnt += 1
        if cnt >= 10:
            break
    return 0

def check_dir(out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

def convert_audio_to_spectrogram(src_dir, out_dir, ext):
    check_dir(out_dir)
    for root, dirs, files in os.walk(src_dir):
        adhi = 1
        for file in files:
            adhi += 1
            if adhi%3000 == 0:
                print(adhi)
            if file.endswith(ext):
                save_path = os.path.join(out_dir, os.path.splitext(file)[0] + '.png')
                if not os.path.exists(save_path):
                    sig, fs = librosa.load(os.path.join(root, file))
                    pylab.figure(figsize=(1.28, 0.96), dpi=100)
                    # Remove axis
                    pylab.axis('off')
                    # Remove the white edge
                    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
                    s = librosa.feature.melspectrogram(y=sig, sr=fs)
                    librosa.display.specshow(librosa.power_to_db(s, ref=np.max))
                    save_path = os.path.join(out_dir, os.path.splitext(file)[0] + '.png')
                    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
                    pylab.close()


src_dir = "C:/Users/adhis/Desktop/Scalable Heavy Data stuff/final_submission/"
func2(src_dir = src_dir+'f2 aud/',out_dir = src_dir+'f2 aud sep/')
convert_audio_to_spectrogram(src_dir=src_dir+"f2 aud sep/",out_dir=src_dir+'f2 spects', ext='mp3')