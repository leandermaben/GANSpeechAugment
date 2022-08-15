import torch
import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
import multiprocessing
import util.util as util
from PIL import Image
import subprocess
from itertools import chain
from collections import OrderedDict
import math
from joblib import Parallel, delayed
import numpy as np

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
    mag_spec, phase, sr = util.extract(filepath, sr=8000, energy=1.0, state = state)
    components = split_and_save(mag_spec, pow=power, state = state, channels = channels)

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
        parser.set_defaults(preprocess='resize',load_size=128)
        
        return parser

    def __init__(self,opt):
        BaseDataset.__init__(self,opt)
        print(f'Creating Dataset with {opt.split}')
        self.dir_A = os.path.join(opt.dataroot,opt.class_ids[0],opt.split)
        self.dir_B = os.path.join(opt.dataroot,opt.class_ids[1],opt.split)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        assert len(self.A_paths) == len(self.B_paths), 'Both domains should have same number of samples for aligned Dataset.'

        self.spec_power = opt.spec_power
        self.energy = opt.energy
        self.phase = opt.phase
        self.channels = 1
        self.num_cores = multiprocessing.cpu_count()

        if("passcodec" in opt.preprocess):
            print("#"*25)
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
        print(self.clean_specs_len,self.noisy_specs_len)
        assert self.noisy_specs_len == len(self.noisy_spec_paths)
        assert self.clean_specs_len == self.noisy_specs_len
        del self.no_comps_noisy


    def __getitem__(self,index):
        A_path = self.clean_spec_paths[index]
        A_img = self.clean_specs[index]
        
        B_path = self.noisy_spec_paths[index]
        B_img = self.noisy_specs[index]

        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale= True)
        B_transform = get_transform(self.opt, transform_params, grayscale= True)
        A = A_transform(A_img)
        B = B_transform(B_img)

        if (self.phase).lower() == 'train':
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_comps': self.clean_comp_dict[A_path]}


    def __len__(self):
        return self.noisy_specs_len