import numpy as np
import librosa 
from sklearn.decomposition import NMF
import sklearn.decomposition.nmf as nmf
d_w = 2048
d_h = 512
nc  = 6
beta = 1
max_iter = 1000

def extractTemplate(y, w=d_w, h=d_h, n_components=nc):
    model = NMF(n_components=n_components, max_iter=max_iter, beta=beta)
    S = librosa.core.stft(y, n_fft=w, hop_length=h)
    model.fit_transform(np.abs(S).T)
    components = model.components_.T
    #components, activation = librosa.decompose.decompose(np.abs(S), n_components=3)
    return components

def extractActivation(y, W, w=d_w, h=d_h):
    """
        important : in sklearn, H is "dictionary", while W is "activation".
        but in our case, W is "dictionary". So we have to pass W as H into sklearn
    """
    S = librosa.core.stft(y, n_fft=w, hop_length=h)
    activation, components, n_iter = nmf.non_negative_factorization(X=np.abs(S.T), H=W.T, update_H=False, n_components=W.shape[1], beta=beta, max_iter=max_iter)
    return activation.T


def mapframe(i, d_h, fs=44100.0):
    return int(float(i / 1000.0) * fs / h) + 1

def cons_Activation(score_inf, H, h_size=d_h, fs=44100.0, time_length=5.0):
    """
        1 vio : 55 - 99
        2 cla : 50 - 89
    """   
    mask  = np.zeros(H.shape) 
    bound = nc * 45

    for inf in score_inf:

        """
            columns here represent time frame
        """
        r_1 = mapframe(inf[0])
        r_2 = mapframe(inf[1])
        note = inf[2]
        inst = inf[3]
        
        if inst == 1 :
            """
                case : vio (range(0, bound))
            """
            begin = note - 55
            u = begin * nc
            l = u + nc
            mask[u:l, r_1:r_2] = 1
        else :
            """
                case : cla (range(bound, H.shape[0]))
            """ 
            begin = note - 50
            u = begin * nc + bound
            l = u + nc
            mask[u:l, r_1:r_2] = 1
    
    return H * mask
