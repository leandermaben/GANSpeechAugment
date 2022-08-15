import argparse
import os
import cv2
import numpy as np
import torch
import librosa
import torchaudio
import pickle
from util.util import save_pickle,load_pickle_file

RESULTS_PATH_DEFAULT = '/content/Pix2Pix-VC/results/noise_pix2pix/test_latest/images'
SAVE_PATH_DEFAULT = '/content/Pix2Pix-VC/data_cache/results' 
CACHE_DEFAULT = '/content/Pix2Pix-VC/data_cache'
SAMPLING_RATE = 22050

def save_results_as_audio_and_spec(real_a,real_b,fake_b,image_path,save_dir):
    image_path = os.path.basename(image_path)
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    ##TODO:Add to arguments
    padding = load_pickle_file(os.path.join(CACHE_DEFAULT,'noisy','meta','noisy_padding.pickle'))
    clean_stats = np.load(os.path.join(CACHE_DEFAULT,'clean','meta','clean_stat.npz'))
    noisy_stats = np.load(os.path.join(CACHE_DEFAULT,'noisy','meta','noisy_stat.npz'))
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir,'images'))
        os.makedirs(os.path.join(save_dir,'audio'))
    imgs = (real_a,real_b,fake_b)
    suffs = ('real_a','real_b','fake_b')
    os.makedirs('/content/Pix2Pix-VC/demo',exist_ok=True)
    for img,suff in zip(imgs,suffs):
        img=img.squeeze()
        img=img.cpu().numpy()
        key = image_path[:32]
        top_pad,right_pad = padding[key]
        spec = img[top_pad:,0:img.shape[1]-right_pad]
        if suff =='real_b' or suff=='fake_b':
            spec=spec*noisy_stats['std']+noisy_stats['mean']
        else:
            spec=spec*clean_stats['std']+clean_stats['mean']
        save_pickle(spec,os.path.join(save_dir,'images',image_path[:32]+suff+'.pickle'))
        spec =torch.tensor(spec,dtype=torch.float32)
        rev = vocoder.inverse(spec.unsqueeze(0)).cpu().detach()
        torchaudio.save(os.path.join(save_dir,'audio',image_path[:32]+suff+'.wav'), rev, sample_rate=SAMPLING_RATE, bits_per_sample=32)



def get_audio(data_root,padding_path,save_path):
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
    padding = load_pickle_file(padding_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in os.listdir(data_root):
        if file[-4:]!= '.png':
            continue
        img_padded = np.array(cv2.imread(os.path.join(data_root,file),0))
        key = file[:32]
        top_pad,right_pad = padding[key]
        img = img_padded[top_pad:,0:img_padded.shape[1]-right_pad]
        img =torch.tensor(img,dtype=torch.float32)
        rev = vocoder.inverse(img.unsqueeze(0)).cpu().detach()
        torchaudio.save(os.path.join(save_path,file[:-4]+'.wav'), rev, sample_rate=SAMPLING_RATE, bits_per_sample=16)
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate Audiofiles from spectrograms')
    parser.add_argument('--results_path', dest = 'results_path', type=str, default=RESULTS_PATH_DEFAULT, help="Path to results folder")
    parser.add_argument('--save_path', dest='save_path', type=str, default=SAVE_PATH_DEFAULT, help="Directory to save Audio files.")
    parser.add_argument('--data_cache', dest='data_cache', type=str, default=CACHE_DEFAULT, help="Path to data cache")
    args = parser.parse_args()
    get_audio(args.results_path,args.padding_path,args.save_path)