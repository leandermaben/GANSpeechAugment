"""
This module implements a custom verion of Pytorch's Dataset class.
It is adapted to suit audio data.
"""

import torch
import torch.utils.data as data
import os
from data.base_dataset import get_params, get_transform, BaseDataset
from data.audio_folder import make_dataset_audio
import multiprocessing
import util.util as util
from PIL import Image
import random
import subprocess
from itertools import chain
from collections import OrderedDict
import math
from joblib import Parallel, delayed
import numpy as np
import json

"""
Heavily borrows from https://github.com/shashankshirol/GeneratingNoisySpeechData
"""

#Loading defaults

with open('defaults.json','r') as f:
    defaults = json.load(f)


def split_and_save(mag_spec, phase_spec, pow=1.0, state = "Train", channels = 1, use_phase=False):
    """
        Info: Takes a spectrogram, splits it into equal parts; uses median padding to achieve this.
        Parameters:
            mag_spec - Magnitude Spectrogram
            phase_spec - Phase Spectrogram
            pow - value to raise the magnitude spectrogram by
            state - Decides how the components are returned
            use_phase - Decides if phase spectrograms should be returned

        Modified by: Leander Maben
    """


    fix_w = defaults['fix_w']  # because we have 129 n_fft bins; this will result in 129x128 spec components
    orig_shape = mag_spec.shape # mag_spec and phase_spec have same dimensions

    #### adding the padding to get equal splits
    w = orig_shape[1]
    mod_fix_w = w % fix_w
    extra_cols = 0
    if(mod_fix_w != 0):
        extra_cols = fix_w - mod_fix_w
        
    #making padding by repeating same audio (takes care of edge case where actual data < padding columns to be added)
    num_wraps = math.ceil(extra_cols/w)
    temp_roll_mag = np.tile(mag_spec, num_wraps)
    padd_mag=temp_roll_mag[:,:extra_cols]
    mag_spec = np.concatenate((mag_spec, padd_mag), axis=1)

    temp_roll_phase = np.tile(phase_spec, num_wraps)
    padd_phase=temp_roll_phase[:,:extra_cols]
    phase_spec = np.concatenate((phase_spec, padd_phase), axis=1)
    ####

    spec_components = []

    mag_spec = util.power_to_db(mag_spec**pow)

    X_mag, _, _ = util.scale_minmax(mag_spec, 0, 255)
    X_phase, _, _ = util.scale_minmax(phase_spec, 0, 255)
    X_mag = np.flip(X_mag, axis=0)
    X_phase = np.flip(X_phase, axis=0)
    np_img_mag = X_mag.astype(np.uint8)
    np_img_phase = X_phase.astype(np.uint8)

    curr = [0]
    while(curr[-1] < w):
        temp_spec_mag = np_img_mag[:, curr[-1]:curr[-1] + fix_w]
        temp_spec_phase = np_img_phase[:, curr[-1]:curr[-1] + fix_w]
        #rgb_im = util.to_rgb(temp_spec, chann = channels)
        mag_img = Image.fromarray(temp_spec_mag)
        phase_img = Image.fromarray(temp_spec_phase)
        if use_phase:
            spec_components.append([mag_img,phase_img])
        else:
            spec_components.append(mag_img)

        curr.append(curr[-1] + fix_w)

    if(state == "Train"):
        return spec_components if extra_cols == 0 else spec_components[:-1]  # No need to return the component with padding.
    else:
        return spec_components  # If in "Test" state, we need all the components


def processInput(filepath, power, state, channels, use_phase):
    mag_spec, phase, sr = util.extract(filepath, sr=defaults["sampling_rate"], energy=1.0, state = state)
    components = split_and_save(mag_spec, phase, pow=power, state = state, channels = channels, use_phase=use_phase)

    return components


def countComps(sample):
    return len(sample)

class AudioDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--class_ids', dest='class_ids', type=str, default=['clean','noisy'], help='class IDS of the two domains.')
        parser.add_argument('--spec_power', dest='spec_power', type=float, default=1.0, help='Number to raise spectrogram by.')
        parser.add_argument('--energy', dest='energy', type=float, default=1.0, help='to modify the energy/amplitude of the audio-signals')
        parser.set_defaults(preprocess='resize',load_size_h=defaults["load_size_h"], load_size_w=defaults["load_size_w"], crop_size=min(defaults["load_size_h"],defaults["load_size_w"])) #TODO Change crop to h and w

        return parser

    def __init__(self,opt):
        BaseDataset.__init__(self,opt)
        print(f'Initializing Dataset for {opt.phase} mode.')
        self.dir_A = os.path.join(opt.dataroot,opt.class_ids[0],opt.phase)
        self.A_paths = sorted(make_dataset_audio(self.dir_A, opt.max_dataset_size))
        
        if not opt.single_direction:
            self.dir_B = os.path.join(opt.dataroot,opt.class_ids[1],opt.phase)
            self.B_paths = sorted(make_dataset_audio(self.dir_B, opt.max_dataset_size))


        self.opt=opt
        self.spec_power = opt.spec_power
        self.energy = opt.energy
        self.phase = opt.phase
        self.channels = 1
        self.num_cores = multiprocessing.cpu_count()
        self.data_load_order = opt.data_load_order
        #self.max_mask_len = opt.max_mask_len

        if self.opt.use_mask:
            print('########## Using Mask ##########')

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
        self.clean_specs = Parallel(n_jobs=self.num_cores, prefer="threads")(delayed(processInput)(i, self.spec_power, self.phase, self.channels, self.opt.use_phase) for i in self.A_paths)
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
        if not self.opt.single_direction:
            self.noisy_specs = Parallel(n_jobs=self.num_cores, prefer="threads")(delayed(processInput)(i, self.spec_power, self.phase, self.channels, self.opt.use_phase) for i in self.B_paths)
            self.no_comps_noisy = Parallel(n_jobs=self.num_cores, prefer="threads")(delayed(countComps)(i) for i in self.noisy_specs)

            self.noisy_spec_paths = []
            self.noisy_comp_dict = OrderedDict()
            for nameB, countB in zip(self.B_paths, self.no_comps_noisy):
                self.noisy_spec_paths += [nameB] * countB
                self.noisy_comp_dict[nameB] = countB
            self.noisy_specs = list(chain.from_iterable(self.noisy_specs))
            self.noisy_specs_len = len(self.noisy_specs)
            assert self.noisy_specs_len == len(self.noisy_spec_paths)
            del self.no_comps_noisy
        else:
            self.noisy_specs=self.clean_specs
            self.noisy_specs_len = self.clean_specs_len
            self.noisy_spec_paths = self.clean_spec_paths
            self.noisy_comp_dict = self.clean_comp_dict

    def get_mask(self,A):
        # Generating mask (for filling in frames) if required 
        if self.opt.phase == 'train':
            mask_size = np.random.randint(0,self.opt.max_mask_len)
            start = np.random.randint(0,A.size(1)-mask_size)
            end = start+mask_size
            mask = torch.ones(1,A.size(1),A.size(2))
            mask[:,:,start:end] = 0
        else:
            mask = torch.ones(1,A.size(1),A.size(2))
        return mask


    def __getitem__(self,index):
        
        index_A = index % self.clean_specs_len
        A_path = self.clean_spec_paths[index_A]  # make sure index is within then range
        A_img = self.clean_specs[index_A]
        if self.opt.use_phase:
            transform_params_A = get_params(self.opt, A_img[0].size)
            A_transform = get_transform(self.opt, transform_params_A, grayscale= True)
            A = torch.cat([A_transform(A_img[0]),A_transform(A_img[1])], dim=0) # Appending magnitude and phase components along the channel dimension
        else:
            transform_params_A = get_params(self.opt, A_img.size)
            A_transform = get_transform(self.opt, transform_params_A, grayscale= True)
            A = A_transform(A_img)

        if self.data_load_order == 'aligned' or self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.noisy_specs_len
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.noisy_specs_len - 1)
        B_path = self.noisy_spec_paths[index_B]
        B_img = self.noisy_specs[index_B]
        if self.opt.use_phase:
            transform_params_B = get_params(self.opt, B_img[0].size)
            B_transform = get_transform(self.opt, transform_params_B, grayscale= True)
            B = torch.cat([B_transform(B_img[0]),B_transform(B_img[1])], dim=0) # Appending magnitude and phase components along the channel dimension
        else:
            transform_params_B = get_params(self.opt, B_img.size)
            B_transform = get_transform(self.opt, transform_params_B, grayscale= True)
            B = B_transform(B_img)

        if (self.phase).lower() == 'train':
            if self.opt.use_mask:
                A_mask = self.get_mask(A)
                B_mask = self.get_mask(B)
                return {'A': A, 'B': B, 'A_mask':A_mask, 'B_mask':B_mask, 'A_paths': A_path, 'B_paths': B_path}
            else:
                return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        else:
            if self.opt.use_mask:
                A_mask = self.get_mask(A)
                B_mask = self.get_mask(B)
                return {'A': A, 'B': B, 'A_mask':A_mask, 'B_mask':B_mask, 'A_paths': A_path, 'B_paths': B_path, 'A_comps': self.clean_comp_dict[A_path], 'B_comps':self.noisy_comp_dict[B_path]}
            else:
                return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_comps': self.clean_comp_dict[A_path], 'B_comps':self.noisy_comp_dict[B_path]}

    def get_A_len(self):
        return self.clean_specs_len

    def get_B_len(self):
        return self.noisy_specs_len

    def __len__(self):
        return max(self.clean_specs_len,self.noisy_specs_len)