import NMF
import basic
import advance
import librosa
import numpy as np

def estimateValidSet(path, vio_W, cla_W, score_inf=None):
    for i in range(0, 5): 
        p = path + '0'
        x = str(i + 1)
        print('round : ' + x)

        valid_v = librosa.load(p + x + '_vio.wav', 44100)[0]
        valid_c = librosa.load(p + x + '_cla.wav', 44100)[0]
        valid_m = librosa.load(p + x + '_mix.wav', 44100)[0]
        sc = None

        if score_inf is not None:
            sc = score_inf[i]

        estimate(valid_m, vio_W, cla_W, valid_v, valid_c, x, sc)

def estimate(valid, v_W, c_W, valid_v, valid_c, r, score_inf=None):
    fn  = 'est/0' + r
    W = np.append(v_W, c_W, axis=1)
    H = NMF.extractActivation(valid, W)
    t_num = v_W.shape[1]
    
    if score_inf is not None :
        print('apply score inf')
        H = NMF.cons_Activation(score_inf, H) 

    v_H = H[0:t_num]
    c_H = H[t_num:H.shape[0]]
   
    o_v, o_c = reconstruct(valid, v_W, v_H, c_W, c_H)
    
    librosa.output.write_wav(fn + '_vio_est.wav', o_v, 44100)
    librosa.output.write_wav(fn + '_cla_est.wav', o_c, 44100)
    
    valid_v = valid_v[0:220160]
    valid_c = valid_c[0:220160]
    
    sdr, sir, sar, perm = basic.evalBSS(np.array([valid_v, valid_c]), np.array([o_v, o_c]))
   
    print(sdr)
    print(sir)
    print(sar)

def reconstruct(y, a_W, a_H, b_W, b_H):
    a = np.dot(a_W, a_H)
    b = np.dot(b_W, b_H)
    D = librosa.core.stft(y, n_fft=NMF.d_w, hop_length=NMF.d_h)
    mag, phase = librosa.magphase(D)
    
    mask_b = 1
    rec_a = a * phase
    rec_b = b * phase
    #mask_b = b**1 / (a**1 + b**1)
    #rec_b = b * mask_b * phase
    #np.abs(D) * mask_b * phase

    o_a = librosa.core.istft(rec_a, win_length=NMF.d_w, hop_length=NMF.d_h)
    o_b = librosa.core.istft(rec_b, win_length=NMF.d_w, hop_length=NMF.d_h)
    
    return o_a, o_b

