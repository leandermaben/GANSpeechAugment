"""
Defines the util functions associated with the cycleGAN VC pipeline.
"""

import io
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchaudio
from torchvision.transforms import ToTensor

import librosa
import librosa.display

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import os
import pickle
import pyloudnorm as pyln
import soundfile as sf
import json

#Loading defaults

with open('defaults.json','r') as f:
    defaults = json.load(f)

"""
Borrows from https://github.com/shashankshirol/GeneratingNoisySpeechData
"""


STANDARD_LUFS = -23.0


def extract(filename, sr=None, energy = 1.0, hop_length = 64, state = None):
    """
        Extracts spectrogram from an input audio file
        Arguments:
            filename: path of the audio file
            n_fft: length of the windowed signal after padding with zeros.
    """
    data, sr = librosa.load(filename, sr=sr)
    data *= energy

    ##Normalizing to standard -23.0 LuFS
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(data)
    data = pyln.normalize.loudness(data, loudness, target_loudness = STANDARD_LUFS)
    ##################################################

    comp_spec = librosa.stft(data, n_fft=256, hop_length = hop_length, window='hamming')

    mag_spec, phase = librosa.magphase(comp_spec)

    phase_in_angle = np.angle(phase)
    return mag_spec, phase_in_angle, sr

def power_to_db(mag_spec):
    return librosa.power_to_db(mag_spec)

def db_to_power(mag_spec):
    return librosa.db_to_power(mag_spec)

def denorm_and_numpy(inp_tensor):
    inp_tensor = inp_tensor[0, :, :, :] #drop batch dimension
    inp_tensor = inp_tensor.permute((1, 2, 0)) #permute the tensor from C x H x W to H x W x C (numpy equivalent)
    inp_tensor = ((inp_tensor * 0.5) + 0.5) * 255 #to get back from transformation
    inp_tensor = inp_tensor.numpy().astype(np.uint8) #generating Numpy ndarray
    return inp_tensor

def getTimeSeries(im, img_path, pow, energy = 1.0, state = None ,train_min=None, train_max=None):
    mag_spec, phase, sr = extract(img_path[0], defaults["sampling_rate"], energy, state = state)
    #TODO : Generalize for pow other than 1 used during training
    log_spec = power_to_db(mag_spec)

    h, w = mag_spec.shape
    
    ######Ignoring padding
    fix_w = 128
    mod_fix_w = w % fix_w
    extra_cols = 0
    if(mod_fix_w != 0):
        extra_cols = fix_w - mod_fix_w
        im = im[:, :-extra_cols]
    #########################
    print("im shape (ex. padding) = ", im.shape)
    print("spec shape (original) = ", mag_spec.shape)

    _min, _max = log_spec.min(), log_spec.max()

    if train_min == None:
        train_min = _min
    if train_max == None:
        train_max = _max

    if(len(im.shape) > 2):
        im = np.mean(im, axis=2)
    im = np.flip(im, axis=0)

    im = unscale_minmax(im, float(train_min), float(train_max), 0, 255)
    spec = db_to_power(im)
    spec = np.power(spec, 1. / pow)

    return reconstruct(spec, phase)/energy, sr

def reconstruct(mag_spec, phase):
    """
        Reconstructs frames from a spectrogram and phase information.
        Arguments:
            mag_spec: Magnitude component of a spectrogram
            phase:  Phase info. of a spectrogram
    """
    temp = mag_spec * np.exp(phase * 1j)
    data_out = librosa.istft(temp)
    return data_out

# to convert the spectrogram ( an 2d-array of real numbers) to a storable form (0-255)
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled, X.min(), X.max()


# to get the original spectrogram ( an 2d-array of real numbers) from an image form (0-255)
def unscale_minmax(X, X_min, X_max, min=0.0, max=1.0):
    X = X.astype(np.float)
    X -= min
    X /= (max - min)
    X *= (X_max - X_min)
    X += X_min
    return X

def to_rgb(im, chann):  # converting the image into 3-channel for singan
    if(chann == 1):
        return im
    w, h = im.shape
    ret = np.empty((w, h, chann), dtype=np.uint8)
    ret[:, :, 0] = im
    for i in range(1, chann):
        ret[:, :, i] = ret[:, :, 0]
    return ret
        
    

def decode_melspectrogram(vocoder, melspectrogram, mel_mean, mel_std):
    """Decoded a Mel-spectrogram to waveform using a vocoder.

    Args:
        vocoder (torch.nn.module): Vocoder used to decode Mel-spectrogram
        melspectrogram (torch.Tensor): Mel-spectrogram to be converted
        mel_mean ([type]): Mean of the Mel-spectrogram for denormalization
        mel_std ([type]): Standard Deviations of the Mel-spectrogram for denormalization

    Returns:
        torch.Tensor: decoded Mel-spectrogram
    """
    denorm_converted = melspectrogram * mel_std + mel_mean
    rev = vocoder.inverse(denorm_converted.unsqueeze(0))
    return rev


def get_mel_spectrogram_fig(spec, title="Mel-Spectrogram"):
    """Generates a figure of the Mel-spectrogram and converts it to a tensor.

    Args:
        spec (torch.Tensor): Mel-spectrogram
        title (str, optional): Figure name. Defaults to "Mel-Spectrogram".

    Returns:
        torch.Tensor: Figure as tensor
    """
    figure, ax = plt.subplots()
    canvas = FigureCanvas(figure)
    S_db = librosa.power_to_db(10**spec.numpy().squeeze(), ref=np.max)
    img = librosa.display.specshow(S_db, ax=ax, y_axis='log', x_axis='time')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
        
    image = Image.open(buf)
    image = ToTensor()(image)
    
    plt.close(figure)
    return image

def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)

def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)
