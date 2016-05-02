import numpy as np
import librosa 
from sklearn.decomposition import NMF
import sklearn.decomposition.nmf as nmf
d_w = 2048
d_h = 512
model = NMF(n_components=3)

def extractTemplate(y, w=d_w, h=d_h):
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
    activation, components, n_iter = nmf.non_negative_factorization(X=np.abs(S.T), H=W.T, update_H=False, n_components=W.shape[1])
    #activation = m.fit_transform(np.abs(S.T), H=W, update_H=False).T
    #components, activation = librosa.decompose.decompose(np.abs(S), transformer=model, fit=False)
    #return model.components_
    return activation.T


