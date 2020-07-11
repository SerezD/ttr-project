#!/usr/bin/env/python

from os import listdir, path
import librosa
import numpy as np
from scipy.stats import skew
from scipy.fftpack import dct
import pywt

# Dataset (GTZAN)
dataset_path = ".\\Data\\Dataset"

# Each directory in dataset_path is a class (contains audio files)
# Classes: Blues Classical Country Disco Hiphop Jazz Metal Pop Reggae Rock
genres = [d for d in listdir(dataset_path) if path.isdir(path.join(dataset_path,d))]

# Final Feature Vector = ZCR mean and var; 20's MFCC mean, var and skew; 4 Autocorrelation Plot 
feat_vector = np.empty ((0,66))

#DWCH (not in final version)
DWCH = False

for g in genres:

    print(g)

    # every file in this directory-genre (100 samples - 30 sec)
    loc_path = dataset_path + "\\" + g
    files = [f for f in listdir(loc_path) if path.isfile(path.join(loc_path, f))] 

    for audio in files:

        # Audio features vector
        audio_feat = np.empty((1,0))

        # Load Audio
        y, _ = librosa.load(loc_path + "\\" + audio)
      
        # DWCH
        if DWCH:

            # Daubechies wavelet filter db8 with 7 levels of decomposition
            dws = pywt.wavedec(y, 'db8', level = 7)
     
            for lvl in dws:

                #Subband energy = Mean of the absolute value of coefficients 
                sub_energy = np.mean( abs(lvl) )
            
                # Discrete cosine Transform
                lvl = dct(lvl)

                # Compute Histogram of dct
                histogram, _ = np.histogram(lvl)

                # Mean, Variance and skew of histogram
                mean = np.mean(histogram)
                variance = np.var(histogram)
                simm = skew(histogram)

            audio_feat = np.append(audio_feat, [mean, variance, simm, sub_energy])
 
        # Autocorrelation Plot

        # Percussive part of signal
        y_perc = librosa.effects.percussive(y)

        # Onset strenght and Tempogram
        oenv = librosa.onset.onset_strength(y=y_perc, hop_length = 512)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, hop_length = 512)

        # Autocorrelation Plot with normalization
        ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
        ac_global = librosa.util.normalize(ac_global)

        # First Feature = first local minimum of plot (value)
        first_min = 2
        index = -1
        for i in range(len(ac_global)):
            if ac_global[i] < first_min:
                first_min = ac_global[i]
                index = i
            else:
                break
        audio_feat= np.append(audio_feat, [first_min])

        # 2nd - 3d Features = mean and variance of peaks
        peaks = []
        p = -1
        up = True

        for val in ac_global[index : ]:
            if up:
                if val < p:
                    peaks.append(p)
                    up = False 
                p = val

            else:
                if val > p:
                    up = True
                p = val

        audio_feat= np.append(audio_feat, [np.mean(peaks), np.var(peaks)])

        # 4th Feature = mean difference between each peak 
        peaks_2 = []
        p = -1
        up = True

        for val in peaks:
            if up:
                if val < p:
                    peaks_2.append(p)
                    up = False 
                p = val

            else:
                if val > p:
                    up = True
                p = val
        
        diff = []
        for i in range(len(peaks_2)-1):
            diff.append(peaks_2[i]-peaks_2[i+1])

        if len(diff) == 0:
            audio_feat= np.append(audio_feat, [0])
        else: 
            audio_feat= np.append(audio_feat, [np.mean(diff)])

        
        # Compute ZCR (mean and variance)
        zcr = librosa.feature.zero_crossing_rate(y)
        audio_feat= np.append(audio_feat, [np.mean(zcr), np.var(zcr)])
        
        # STFT
        # Magnitude and Phase of spectogram (STFT)
        S, _ = librosa.magphase(librosa.core.stft(y)) 
            
        # Compute MFCC
        S_mel = librosa.feature.melspectrogram(S=S)
        mfcc = librosa.feature.mfcc(S=S_mel, n_mfcc= 20)

        # For each Mfcc's subband
        for band in mfcc:

            # Compute mean, variance and skew
            mfcc_mean = np.mean(band)          
            mfcc_var = np.var(band)
            mfcc_skew = skew(band)

            audio_feat= np.append(audio_feat, [mfcc_mean, mfcc_var, mfcc_skew])
        
        # Update Feature Vector
        feat_vector = np.concatenate( (feat_vector, [audio_feat]), axis = 0)
        
# Once done, save everything 
save_path = ".\\Data\\"
np.save(save_path + "features_vector.npy", feat_vector)