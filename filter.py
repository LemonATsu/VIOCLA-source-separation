import numpy as np
import librosa 
import scipy.signal as signal
import matplotlib.pyplot as plt
path = '../audio/train/vio/'


def butter_filter(cutoff, fs, btype, order):
    nyq = 0.5 * fs
    ncutoff = cutoff / nyq
    print(ncutoff)
    b, a = signal.butter(order, ncutoff, btype=btype, analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_filter(cutoff, fs, btype='lowpass', order=order)
    y = signal.lfilter(b, a, data)
    return y

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_filter(cutoff, fs, btype='highpass', order=order)
    y = signal.lfilter(b, a, data)
    return y

if __name__ == '__main__':
    w = 2048
    h = 1024
    fs = 44100
    rate =  float(1200)/22050
    x = librosa.load(path + 'vio_64.wav', fs)[0]


    print('lowpass')
    o = x
    o = lowpass_filter(x, 1200, fs)
    S = librosa.stft(o, n_fft=w, hop_length=h)
    librosa.display.specshow(librosa.logamplitude(np.abs(S)**2, ref_power=np.max), y_axis='log', x_axis='time')
    plt.show()

