import numpy as np
import librosa
import NMF
import Util as u
import matplotlib.pyplot as plt
import basic
import scipy.signal as signal
import est

t_num = 45 * NMF.nc

def extractAllTemplate(data, n_components=NMF.nc):
    init = False
    for d in data:
        comp = NMF.extractTemplate(d[0][0:70000], n_components=n_components)
        if init == False:
            init = True
            W = comp
            continue
        W = np.append(W, comp, axis=1)
    return W


def showDicSpectr(dic):
    librosa.display.specshow(librosa.logamplitude(np.abs(dic)**2), y_axis='linear', x_axis='frames', n_yticks=20)
    plt.axis([0, 135, 0, 300])
    plt.show()

def showActSpectr(act):
    print('show activation')
    librosa.display.specshow(librosa.logamplitude(np.abs(act)**2), y_axis='frames', x_axis='time')
    plt.show()

def showReconSpectr(rec):
    librosa.display.specshow(librosa.logamplitude(rec**2), y_axis='linear', x_axis='time')
    plt.show()

def R1(valid, W):
    H = NMF.extractActivation(valid, W)
    r = np.dot(W, H)
    o = librosa.core.istft(r, win_length=NMF.d_w, hop_length=NMF.d_h)
    print(H[33])
    
    #showDicSpectr(W)
    #showActSpectr(H)
    """showReconSpectr(r)"""
    librosa.output.write_wav('01_vio.wav', o, 44100)

def R2(valid, W):
    H = NMF.extractActivation(valid, W)
    r = np.dot(W, H)
    o = librosa.core.istft(r, win_length=NMF.d_w, hop_length=NMF.d_h)
    """showDicSpectr(W)
    showActSpectr(H)
    showReconSpectr(r)"""
    librosa.output.write_wav('01_cla.wav', o, 44100)

def R3(path, vio_W, cla_W):
    est.estimateValidSet(path, vio_W, cla_W)

def R4(v_W, c_W):
    path = '../audio/test/'
    fn = '../pred/'
    test_clips = u.readClips(path)
    W = np.append(v_W, c_W, axis=1)
  
    for i in range(0, len(test_clips)):
        p = fn + '0' + str(i + 6)
        H = NMF.extractActivation(test_clips[i][0], W)
        v_H = H[0:t_num]
        c_H = H[t_num:H.shape[0]]
        o_v, o_c = est.reconstruct(test_clips[i][0], v_W, v_H, c_W, c_H)
        librosa.output.write_wav(p + '_vio_est.wav', o_v, 44100)
        librosa.output.write_wav(p + '_cla_est.wav', o_c, 44100)

if __name__ == '__main__':
    print('require')
    path_valid = '../audio/validation/'
    vio_clips = u.readClips('../audio/train/vio/')
    cla_clips = u.readClips('../audio/train/cla/')
    valid_v = librosa.load(path_valid + '01_vio.wav', 44100)[0]
    valid_c = librosa.load(path_valid + '01_cla.wav', 44100)[0]
    valid_m = librosa.load(path_valid + '01_mix.wav', 44100)[0]
    # cla_clips = u.readClips('../audio/train/cla/')
    init = False   
    vio_W = extractAllTemplate(vio_clips)
    cla_W = extractAllTemplate(cla_clips)
    R1(valid_v, vio_W)
    R2(valid_c, vio_W)   
    R3(path_valid, vio_W, cla_W)    

    R4(vio_W, cla_W)

