import est
import basic, os
import require
import librosa
import numpy as np
import matplotlib.pyplot as plt
import NMF
import require as r
import Util as u

def listParser(list):
    result = []
    for string in list:
        x = string.split('\t')
        for i in range(0, 4):
            x[i] = int(x[i])
            if x[i] > 5000:
                x[i] = 5000
        result.append(x)

    return result

def readNote(path):
    list = []
    
    for file in os.listdir(path):   
        if file.endswith('.txt'):
            print(file)
            with open(path + file) as f:
                list.append(listParser(f.readlines()))
    print(list[0])
    return list

if __name__ == '__main__':
    path = '../audio/validation/'
    score_path = '../score-info/'
    score_inf = readNote(score_path)
    vio_clips = u.readClips('../audio/train/vio/')
    cla_clips = u.readClips('../audio/train/cla/')
    vio_W = r.extractAllTemplate(vio_clips)
    cla_W = r.extractAllTemplate(cla_clips)

    est.estimateValidSet(path, vio_W, cla_W, score_inf)    

