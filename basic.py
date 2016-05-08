import mir_eval
import librosa
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import NMF

fs = 44100

def B1():
    fs = 44100
    path = '../audio/validation/'
    a = librosa.load(path + '01_vio.wav', fs)[0]
    b = librosa.load(path + '01_cla.wav', fs)[0]
    c = librosa.load(path + '01_mix.wav', fs)[0]
    n = np.random.randn(len(a)) 
    print(a.shape)
    #print(evalSDR(np.array([c, c])/2, np.array([a, b]))) 
    # correct one
    print("SDR [c;c]/2, [a;b]")
    print(evalSDR(np.array([a, b]), np.array([c, c])/2))
    print("SDR [a;b], [a;b]")
    print(evalSDR(np.array([a, b]), np.array([a, b])))
    print("SDR [b;a], [a;b]")
    print(evalSDR(np.array([a, b]), np.array([b, a])))
    print("SDR [2a;2b], [a;b]")
    print(evalSDR(np.array([a, b]), np.array([2*a, 2*b])))
    print("SDR (a+0.01*n), a")
    print(evalSDR(a, (a + 0.01*n)))
    print("SDR (a+0.1*n), a")
    print(evalSDR(a, (a + 0.1*n)))
    print("SDR (a+n), a")
    print(evalSDR(a, (a + n)))
    print("SDR (a+0.01*b), a")
    print(evalSDR(a, (a + 0.01*b)))
    print("SDR (a+0.1*b), a")
    print(evalSDR(a, (a + 0.1*b)))
    print("SDR (a+b), a")
    print(evalSDR(a, (a + b)))


def B2(): 
    path = '../audio/train/vio/'
    vio_64 = librosa.load(path + 'vio_64.wav', fs)[0]
    

def B3():
    w = 2048
    h = 1024
    path = '../audio/train/vio/'
    
    vio_64 = librosa.load(path + 'vio_64.wav', fs)[0][0:61000]
    vio_88 = librosa.load(path + 'vio_88.wav', fs)[0][0:61000]
    cla_64 = librosa.load('../audio/train/cla/cla_64.wav', fs)[0][0:61000]
   
    S_1 = NMF.extractTemplate(vio_64)
    S_2 = NMF.extractTemplate(vio_88)
    S_3 = NMF.extractTemplate(cla_64)
 
    librosa.display.specshow(S_1, y_axis='cqt_note', x_axis='frames', n_yticks=180)
    plt.axis([0, 2, 0, 100])
    plt.show()

    librosa.display.specshow(S_2, y_axis='cqt_note', x_axis='frames', n_yticks=10)
    #plt.axis([0, 2, 0, 100])
    plt.show()
    
    librosa.display.specshow(S_3, y_axis='cqt_note', x_axis='frames', n_yticks=180)
    plt.axis([0, 2, 0, 100])
    plt.show()

    S_1 = librosa.core.istft(S_1)
    librosa.display.waveplot(S_1, x_axis='time')
    plt.show()
    S_2 = librosa.core.istft(S_2)
    librosa.display.waveplot(S_2, x_axis='time')
    plt.show()
def evalSDR(ref, est):
    
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(ref, est)
    print(perm)
    return sdr

def evalBSS(ref, est):
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(ref, est)
    return sdr, sir, sar, perm

if __name__ == '__main__':

    print('Basic 1 : start')
    B1()
    print('Basic 1 : end')

    print('Basic 2 : start')
    B2()
    print('Basic 2 : end')


    print('Basic 3 : start')
    B3()
    print('Basic 3 : end')

