import librosa
import os
import numpy as np
import pyloudnorm as pyln
from scipy import signal
from pydub import AudioSegment
import scipy.io.wavfile as wav
import math
import soundfile as sf
import shutil
import pandas as pd
import json
 
#Loading defaults

with open('defaults.json','r') as f:
    defaults = json.load(f)


def stft(audio,n_fft,overlap):
    """
    Perform stft on audio and return magnitude spectrogram.

    Created by Leander Maben.
    """
    comp_spec = librosa.stft(audio,n_fft=n_fft,win_length=n_fft,center=False,hop_length=int(n_fft * (1.0 - overlap)))
    mag_spec = np.abs(comp_spec)
    return mag_spec

def safe_log(x, eps=1e-5):
    safe_x = np.where(x <= eps, eps, x)
    return np.log(safe_x)


def normalize(sig1, sig2):
    """sig1 is the ground_truth file
       sig2 is the file to be normalized"""

    def get_mediainfo(sig):
        rate, data = wav.read(sig)
        bits_per_sample = np.NaN
        if(data.dtype == 'int16'):
            bits_per_sample = 16
        elif(data.dtype == 'int32'):
            bits_per_sample = 32

        return rate, bits_per_sample

    sample_rate1, bits_per_sample_sig1 = get_mediainfo(sig1)
    sample_rate2, bits_per_sample_sig2 = get_mediainfo(sig2)

    ## bps and sample rate must match
    assert bits_per_sample_sig1 == bits_per_sample_sig2
    assert sample_rate1 == sample_rate2

    def match_target_amplitude(sound, target):
        change = target - sound.dBFS
        return sound.apply_gain(change)

    sound1 = AudioSegment.from_wav(sig1)
    sound2 = AudioSegment.from_wav(sig2)

    ## Matching loudness
    sound2 = match_target_amplitude(sound2, sound1.dBFS)

    ## getting it back to librosa form
    samples1 = sound1.get_array_of_samples()
    data1 = np.array(samples1).astype(np.float32)/(2**(bits_per_sample_sig1 - 1))

    samples2 = sound2.get_array_of_samples()
    data2 = np.array(samples2).astype(np.float32)/(2**(bits_per_sample_sig2 - 1))

    return data1, data2, sample_rate1



def compute_mssl(file1,file2,n_ffts, mag_weight=defaults["mssl_mag_weight"], logmag_weight=defaults["mssl_logmag_weight"]):
    loss = 0

    _, aud_1 = wav.read(file1)
    _, aud_2 = wav.read(file2)
    if(np.sum(aud_1.astype(float)**2) > np.sum(aud_2.astype(float)**2)):
        file1, file2 = file2, file1
        
    data1, data2, sr = normalize(sig1=file1, sig2=file2)

    data1, data2 = time_and_energy_align(data1,data2, sr)

    for n_fft in n_ffts:
        spec1 =stft(data1, n_fft, defaults["mssl_overlap"])
        spec2 =stft(data2, n_fft, defaults["mssl_overlap"])
        if mag_weight > 0:
            loss += mag_weight * np.mean(np.abs(spec1-spec2))
        if logmag_weight > 0:
            loss+=logmag_weight * np.mean(np.abs(safe_log(spec1)-safe_log(spec2)))
    return loss

def AddNoiseFloor(data):
    frameSz = defaults["fix_w"]
    noiseFloor = (np.random.rand(frameSz) - 0.5) * 1e-5
    numFrame = math.floor(len(data)/frameSz)
    st = 0
    et = frameSz-1

    for i in range(numFrame):
        if(np.sum(np.abs(data[st:et+1])) < 1e-5):
            data[st:et+1] = data[st:et+1] + noiseFloor
        st = et + 1
        et += frameSz

    return data


def time_and_energy_align(data1, data2, sr):
    nfft = defaults["nfft"]
    hop_length = defaults["align_hop"]  # hop_length = win_length or frameSz - overlapSz
    win_length = defaults["align_win_len"]

    ##Adding small random noise to prevent -Inf problem with Spec
    data1 = AddNoiseFloor(data1)
    data2 = AddNoiseFloor(data2)

    ##Pad with silence to make them equal
    zeros = np.zeros(np.abs((len(data2) - len(data1))), dtype=float)
    padded = -1
    if(len(data1) < len(data2)):
        data1 = np.append(data1, zeros)
        padded = 1
    elif(len(data2) < len(data1)):
        data2 = np.append(data2, zeros)
        padded = 2
    
    
    # Time Alignment
    # Cross-Correlation and correction of lag using the spectrograms
    spec1 = abs(librosa.stft(data1, n_fft=nfft, hop_length=hop_length,
                             win_length=win_length, window='hamming'))
    spec2 = abs(librosa.stft(data2, n_fft=nfft, hop_length=hop_length,
                             win_length=win_length, window='hamming'))
    energy1 = np.mean(spec1, axis=0)
    energy2 = np.mean(spec2, axis=0)
    n = len(energy1)

    corr = signal.correlate(energy2, energy1, mode='same') / np.sqrt(signal.correlate(energy1,
                                                                                      energy1, mode='same')[int(n/2)] * signal.correlate(energy2, energy2, mode='same')[int(n/2)])
    delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n).round(decimals=6)

    #print(np.argmax(corr) - corr.size//2) no. of samples to move

    delay = delay_arr[np.argmax(corr)]
    print('y2 lags by ' + str(delay) + ' to y1')

    if(delay*sr < 0):
        to_roll = math.ceil(delay*sr)
    else:
        to_roll = math.floor(delay*sr)

    # correcting lag
    # if both signals were the same length, doesn't matter which one was rolled
    if(padded == 1 or padded == -1):
        data1 = np.roll(data1, to_roll)
    elif(padded == 2):
        data2 = np.roll(data2, -to_roll)

    #Plot Cross-correlation vs Lag; for debugging only;
    """ plt.figure()
    plt.plot(delay_arr, corr)
    plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show() """

    # Energy Alignment

    data1 = data1 - np.mean(data1)
    data2 = data2 - np.mean(data2)

    sorted_data1 = -np.sort(-data1)
    sorted_data2 = -np.sort(-data2)

    L1 = math.floor(0.01*len(data1))
    L2 = math.floor(0.1*len(data1))

    gain_d1d2 = np.mean(np.divide(sorted_data1[L1:L2+1], sorted_data2[L1:L2+1]))

    #Apply gain
    data2 = data2 * gain_d1d2

    return data1, data2

def main(source_dir=defaults["test_source"],results_dir=defaults["test_results"], use_gender=defaults["use_gender_test"]):

    if use_gender:
        annotations = {}
        anno_csv = pd.read_csv(defaults["annotations"])
        for i in range(len(anno_csv)):
            row=anno_csv.iloc[i]
            annotations[row['file']]=row['gender']

    #Checking for sample rates
    file_0 = os.listdir(source_dir)[0]
    file1 = os.path.join(source_dir,file_0)
    file2 = os.path.join(results_dir,file_0)
    _,file1_rate = librosa.load(file1, sr=None)
    _,file2_rate = librosa.load(file2, sr=None)

    if file1_rate!=file2_rate:
        ## Storing original audios in a new temp cache with desired sample_rate
        TEMP_CACHE = defaults["metrics_temp_cache"]
        os.makedirs(TEMP_CACHE)
        for file in os.listdir(source_dir):
            file1 = os.path.join(source_dir,file)
            loaded_file,_ = librosa.load(file1, sr=file2_rate)
            sf.write(os.path.join(TEMP_CACHE,file), loaded_file, file2_rate, 'PCM_16')
    else:
        TEMP_CACHE = source_dir
    
    male_loss = []
    female_loss = []
    total_loss =[]

    for file in os.listdir(source_dir):
        file1 = os.path.join(TEMP_CACHE,file)
        file2 = os.path.join(results_dir,file)

        loss = compute_mssl(file1,file2,[2048, 1024, 512, 256, 128, 64])

        if use_gender:
            if annotations[file] == 'M':
                male_loss.append(loss)

            else:
                female_loss.append(loss)
        else:
            total_loss.append(loss)

    if use_gender:    
        total_loss = np.concatenate((male_loss,female_loss))

        total_mean = total_loss.mean()
        total_std = total_loss.std()
        male_mean = np.mean(male_loss)
        male_std = np.std(male_loss)
        female_mean = np.mean(female_loss)
        female_std = np.std(female_loss)
        if TEMP_CACHE!=source_dir:
            shutil.rmtree(TEMP_CACHE)
        return total_mean, total_std, male_mean, male_std, female_mean, female_std
    else:
        total_mean = np.mean(total_loss)
        total_std = np.std(total_loss)
        if TEMP_CACHE!=source_dir:
            shutil.rmtree(TEMP_CACHE)
        return total_mean, total_std


    

if __name__ == '__main__':
    print(main())