import torch
import torch.utils.data as data
import os
from dataset.base_functions import get_params, get_transform, make_dataset, make_dataset
import multiprocessing
import mask_cyclegan_vc.utils as util
from PIL import Image
import random
import subprocess
from itertools import chain
from collections import OrderedDict
import math
from joblib import Parallel, delayed
import numpy as np
import json

#Loading defaults

with open('defaults.json','r') as f:
    defaults = json.load(f)
"""
Heavily borrows from https://github.com/shashankshirol/GeneratingNoisySpeechData
"""

def split_and_save(spec, pow=1.0, state = "Train", channels = 1):
    """
        Info: Takes a spectrogram, splits it into equal parts; uses median padding to achieve this.
        Parameters:
            spec - Magnitude Spectrogram
            pow - value to raise the spectrogram by
            phase - Decides how the components are returned
    """

    fix_w = 128  # because we have 129 n_fft bins; this will result in 129x128 spec components
    orig_shape = spec.shape

    #### adding the padding to get equal splits
    w = orig_shape[1]
    mod_fix_w = w % fix_w
    extra_cols = 0
    if(mod_fix_w != 0):
        extra_cols = fix_w - mod_fix_w
        
    #making padding by repeating same audio (takes care of edge case where actual data < padding columns to be added)
    num_wraps = math.ceil(extra_cols/w)
    temp_roll = np.tile(spec, num_wraps)
    padd=temp_roll[:,:extra_cols]
    spec = np.concatenate((spec, padd), axis=1)
    ####

    spec_components = []

    spec = util.power_to_db(spec**pow)
    X, X_min, X_max = util.scale_minmax(spec, 0, 255)
    X = np.flip(X, axis=0)
    np_img = X.astype(np.uint8)

    curr = [0]
    while(curr[-1] < w):
        temp_spec = np_img[:, curr[-1]:curr[-1] + fix_w]
        rgb_im = util.to_rgb(temp_spec, chann = channels)
        img = Image.fromarray(rgb_im)
        spec_components.append(img)
        curr.append(curr[-1] + fix_w)

    if(state == "Train"):
        return spec_components if extra_cols == 0 else spec_components[:-1]  # No need to return the component with padding.
    else:
        return spec_components  # If in "Test" state, we need all the components


def processInput(filepath, power, state, channels):
    mag_spec, phase, sr = util.extract(filepath, sr=defaults["sampling_rate"], energy=1.0, state = state)
    components = split_and_save(mag_spec, pow=power, state = state, channels = channels)

    return components


def countComps(sample):
    return len(sample)

class NoiseDataset(data.Dataset):

    def __init__(self,opt,valid=False):
        self.dir_A = os.path.join(opt.dataroot,opt.class_ids[0],opt.split)
        self.dir_B = os.path.join(opt.dataroot,opt.class_ids[1],opt.split)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        self.opt=opt
        self.valid=valid
        self.spec_power = opt.spec_power
        self.energy = opt.energy
        self.phase = opt.phase
        self.channels = 1
        self.num_cores = multiprocessing.cpu_count()
        self.data_load_order = opt.data_load_order
        self.max_mask_len = opt.max_mask_len

        if("passcodec" in opt.preprocess):
            print("------Passing samples through g726 Codec using FFmpeg------")
            for path in self.A_paths:
                subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', path, '-ar', '8k', '-y', path[:-4] + '_8k.wav'])
                subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', path[:-4] + '_8k.wav', '-acodec', 'g726', '-b:a', '16k', path[:-4] + '_fmt.wav'])
                subprocess.call(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', path[:-4] + '_fmt.wav', '-ar', '8k', '-y', path])
                if(os.name == 'nt'):  # Windows
                    os.system('del ' + path[:-4] + '_fmt.wav')
                    os.system('del ' + path[:-4] + '_8k.wav')
                else:  # Linux/MacOS/BSD
                    os.system('rm ' + path[:-4] + '_fmt.wav')
                    os.system('rm ' + path[:-4] + '_8k.wav')

        #Compute the spectrogram components parallelly to make it more efficient; uses Joblib, maintains order of input data passed.
        self.clean_specs = Parallel(n_jobs=self.num_cores, prefer="threads")(delayed(processInput)(i, self.spec_power, self.phase, self.channels) for i in self.A_paths)
        #self.clean_specs = [processInput(i, self.spec_power, self.phase, self.channels) for i in self.A_paths]

        #calculate no. of components in each sample
        self.no_comps_clean = Parallel(n_jobs=self.num_cores, prefer="threads")(delayed(countComps)(i) for i in self.clean_specs)
        #self.no_comps_clean = [countComps(i) for i in self.clean_specs]
        self.clean_spec_paths = []
        self.clean_comp_dict = OrderedDict()

        for nameA, countA in zip(self.A_paths, self.no_comps_clean):  # Having an OrderedDict to access no. of components, so we can wait before generation to collect all components
            self.clean_spec_paths += [nameA] * countA
            self.clean_comp_dict[nameA] = countA

        ##To separate the components; will treat every component as an individual sample
        self.clean_specs = list(chain.from_iterable(self.clean_specs))
        self.clean_specs_len = len(self.clean_specs)
        assert self.clean_specs_len == len(self.clean_spec_paths)

        del self.no_comps_clean

        self.noisy_specs = Parallel(n_jobs=self.num_cores, prefer="threads")(delayed(processInput)(i, self.spec_power, self.phase, self.channels) for i in self.B_paths)
        self.no_comps_noisy = Parallel(n_jobs=self.num_cores, prefer="threads")(delayed(countComps)(i) for i in self.noisy_specs)
        #self.noisy_specs = [processInput(i, self.spec_power, self.state, self.channels) for i in self.B_paths]
        #self.no_comps_noisy = [countComps(i) for i in self.noisy_specs]
        self.noisy_spec_paths = []
        self.noisy_comp_dict = OrderedDict()
        for nameB, countB in zip(self.B_paths, self.no_comps_noisy):
            self.noisy_spec_paths += [nameB] * countB
            self.noisy_comp_dict[nameB] = countB
        self.noisy_specs = list(chain.from_iterable(self.noisy_specs))
        self.noisy_specs_len = len(self.noisy_specs)
        assert self.noisy_specs_len == len(self.noisy_spec_paths)
        del self.no_comps_noisy
        

    def get_mask(self,A):
        if self.phase == 'train':
            mask_size = np.random.randint(0,self.max_mask_len)
            start = np.random.randint(0,A.size(0)-mask_size)
            end = start+mask_size
            mask = torch.ones_like(A)
            mask[:,start:end] = 0
        else:
            mask = torch.ones_like(A)
        return mask


    def __getitem__(self,index):
        
        index_A = index % self.clean_specs_len
        A_path = self.clean_spec_paths[index_A]  # make sure index is within then range
        A_img = self.clean_specs[index_A]

        transform_params_A = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params_A, grayscale= True)
        A = A_transform(A_img).squeeze(0)
        A_mask = self.get_mask(A)

        if self.data_load_order == 'aligned':   # make sure index is within then range
            index_B = index % self.noisy_specs_len
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.noisy_specs_len - 1)
        B_path = self.noisy_spec_paths[index_B]
        B_img = self.noisy_specs[index_B]
        transform_params_B = get_params(self.opt, B_img.size)
        B_transform = get_transform(self.opt, transform_params_B, grayscale= True)
        B = B_transform(B_img).squeeze(0)
        B_mask = self.get_mask(B)

        if self.valid:
            return A,B

        if (self.phase).lower() == 'train':
            return {'A': A, 'B': B, 'A_mask':A_mask, 'B_mask':B_mask, 'A_paths': A_path, 'B_paths': B_path}
        else:
            return {'A': A, 'B': B, 'A_mask':A_mask, 'B_mask':B_mask, 'A_paths': A_path, 'B_paths': B_path, 'A_comps': self.clean_comp_dict[A_path], 'B_comps':self.noisy_comp_dict[B_path]}

    def get_clean_len(self):
        return self.clean_specs_len
    def get_noisy_len(self):
        return self.noisy_specs_len
    def __len__(self):
        return max(self.noisy_specs_len,self.clean_specs_len)