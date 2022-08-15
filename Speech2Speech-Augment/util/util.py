"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import pickle
import pyloudnorm as pyln
import librosa
import soundfile as sf


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)

def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

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
    inp_tensor = inp_tensor.cpu().numpy().astype(np.uint8) #generating Numpy ndarray
    return inp_tensor

def getTimeSeries(im, img_path, pow, energy = 1.0, state = None):
    mag_spec, phase, sr = extract(img_path[0], 8000, energy, state = state)
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

    if(len(im.shape) > 2):
        im = np.mean(im, axis=2)
    im = np.flip(im, axis=0)

    im = unscale_minmax(im, float(_min), float(_max), 0, 255)
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
