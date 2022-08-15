"""
This module implements functions to fetch data and organize them appropiately for training and testing.
There are several ways of doing this. The mode of transfer can be selected using the command line argument --tranfer_mode.
Following options are available: 'audio','spectrogram','npy','codec', 'additive_noise
Audio mode:
Transfer data from audio_data_path directory to data_cache directory with splits.
Following arguments may be provided: --audio_data_path --source_sub_directories --data_cache --annotations_path --train_percent --test_percent --use_genders
Following directory structure is expected:
-audio_data_path
    -source_sub_directories[0]
        -sample_0
        -....
    -source_sub_directories[1]
        -sample_0
        -sample_1
It is also expected that the audios are present as corresponding pairs in the subdirectories.
get_fileNames function also need to be implemented if the corresponding files pairs do not have the same name.
Spectrogram:
Audio data is converted to spectrograms using MelGAN and saved as pickle files(spectrogram) and npz file (statistical values).
Following arguments may be provided: --audio_data_path --source_sub_directories --size_multiple --data_cache --train_percent --test_percent --sampling_rate
Following directory structure is expected:
-audio_data_path
    -source_sub_directories[0]
        -sample_0
        -....
    -source_sub_directories[1]
        -sample_0
        -sample_1
This method does not produce good quality audios when converting from spectrogram to audio after processing. Hence, it is not recommended.
NPY mode:
Can be used if train and test data are present are separate npy files such that clean_data = npy[:, :, 0] and noisy_data = npy[:,:,1] 
Following arguments may be provided: --npy_train_source --npy_test_source --sampling_rate
Codec mode:
Can be used if clean data is available but noisy data is unavailable.
Noisy data can be generated using codec.
Following arguments may be provided: --codec_clean_path --data_cache --train_percent --test_percent
Additive Noise:
Can be used if a noise file is available and all noise is to be generated from that file.
Following arguments may be provided: --clean_path, --noise_file, --data_cache, --train_speakers, --test_speakers, --train_duration_max, --test_duration_max
"""

import argparse
import shutil
import torch
import tqdm
import os
import numpy as np
import pandas as pd
import math
import cv2
import librosa
import pickle
import matplotlib.pyplot as plt
from util.util import save_pickle
import soundfile as sf
import glob
import json

#Loading defaults

with open('defaults.json','r') as f:
    defaults = json.load(f)

def run(command):
    #print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)



## mel function inspired from https://github.com/GANtastic3/MaskCycleGAN-VC

def mel(wavspath, sr):
    info = {
        'records_count' : 0, #Keep track of number of clips
        'duration' : 0 # Keep track of duration of training set for the speaker
    }
    
    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

    mel_list = list()
    wav_files = os.listdir(wavspath)
    filenames = []
    for file in wav_files:
        wavpath = os.path.join(wavspath,file)
        if wavpath[-4:] != '.wav':
            continue
        wav_orig, _ = librosa.load(wavpath, sr=sr, mono=True)
        spec = vocoder(torch.tensor([wav_orig]))
        #print(f'Spectrogram shape: {spec.shape}')
        if spec.shape[-1] >= 64:    # training sample consists of 64 frames
            mel_list.append(spec.cpu().detach().numpy()[0])
            info['duration']+=librosa.get_duration(filename=wavpath)
            info['records_count']+=1   
            filenames.append(file)       
    return mel_list, filenames, info


def preprocess_dataset_spectrogram(data_path, class_id, sr, cache_folder, train_percent, test_percent, size_multiple,args):
    """Preprocesses dataset of .wav files by converting to Mel-spectrograms.
    Args:
        data_path (str): Directory containing .wav files of the speaker.
        class_id (str): ID of the class.
        cache_folder (str, optional): Directory to hold preprocessed data. Defaults to './cache/'.
        Modified By Leander Maben.
    """

    print(f"Preprocessing data for class: {class_id}.")


    mel_list, filenames, info = mel(data_path, sr)

    if not os.path.exists(os.path.join(cache_folder, class_id)):
        os.makedirs(os.path.join(cache_folder, class_id, 'meta'))
        os.makedirs(os.path.join(cache_folder, class_id, 'train'))
        os.makedirs(os.path.join(cache_folder, class_id, 'val'))
        os.makedirs(os.path.join(cache_folder, class_id, 'test'))
    
    indices = np.arange(0,len(mel_list))
    np.random.seed(0)
    np.random.shuffle(indices)

    train_split = math.floor(train_percent/100*len(mel_list))
    val_split = math.floor(val_percent/100*len(mel_list))
    
    padding ={}

    for phase,(start, end) in zip(['train','val','test'],[(0,train_split),(train_split,train_split+val_split),(train_split+val_split,len(mel_list))]):
        if phase=='train':
            ## Get mean and norm
            train_samples=mel_list[start:end]
            mel_concatenated = np.concatenate(train_samples, axis=1)
            mel_mean = np.mean(mel_concatenated, axis=1, keepdims=True)
            mel_std = np.std(mel_concatenated, axis=1, keepdims=True) + 1e-9
            np.savez(os.path.join(cache_folder, class_id, 'meta', f"{class_id}_stat"),mean=mel_mean,std=mel_std)

        for i in range(start,end):
            filename=filenames[indices[i]]
            img = (mel_list[indices[i]]-mel_mean)/mel_std
            filename=filename[:32]+'.pickle' ## THIS STEP IS SPECIFIC TO THE CURRENT DATASET TO ENSURE A AND B HAVE SAME FILENAMES
            
            ##Padding the image
            freq_len,time_len = img.shape
            top_pad = size_multiple - freq_len % size_multiple if freq_len % size_multiple!=0 else 0
            right_pad = size_multiple - time_len % size_multiple if time_len % size_multiple!=0 else 0
            x_size = time_len+right_pad
            y_size = freq_len+top_pad
            img_padded = np.zeros((y_size,x_size))
            img_padded[-freq_len:,0:time_len] = img

            ## Saving Padding info
            padding[filename[:-7]] = (top_pad,right_pad)

            ##Saving Image
            save_pickle(variable=img_padded,fileName=os.path.join(cache_folder,class_id,phase,filename))
        

    
    save_pickle(variable=padding,fileName=os.path.join(cache_folder, class_id, 'meta', f"{class_id}_padding.pickle"))

    print('#'*25)
    print(f"Preprocessed and saved data for class: {class_id}.")
    print(f"Total duration of dataset for {class_id} is {info['duration']} seconds")
    print(f"Total clips in dataset for {class_id} is {info['records_count']}")
    print('#'*25)



def get_filenames(fileNameA):
    """
    Custom function for this specific dataset.
    It returns the names of corresponding files in the 2 classes along with the common name by which it should be saved.
    Args:
    fileNameA(str) : Filename in the first class
    Created By Leander Maben
    """

    return fileNameA, fileNameA[:32]+'-A.wav', fileNameA[:32]+'.wav'



def transfer_aligned_audio_raw(root_dir,class_ids,data_cache,train_percent,test_percent, use_genders, annotations_path, get_filenames=lambda x:(x,x,x)):
    """
    Transfer audio files to a convinient location for processing with train,test,validation split.
    Important Note: The splitting of data by percent is based on file numbers and not on cummulative duration
    of clips. Moreover, it does not take into the account the number of clips that are discarded for being less than 1 second long.
    Arguments:
    root_dir(str) - Root directory where files of specified classes are present in subdirectories.
    class_id(str) - Current class ID of data objects
    data_cache(str) - Root directory to store data
    train_percent(int) - Percent of data clips in train split
    test_percent(int) - Percent of data clips in test split
    Created By Leander Maben. 
    """

    if use_genders != 'None':
        annotations = {}
        anno_csv = pd.read_csv(annotations_path)
        for i in range(len(anno_csv)):
            row=anno_csv.iloc[i]
            annotations[row['file']]=row['gender']


    for class_id in class_ids:
        os.makedirs(os.path.join(data_cache,class_id,'train'))
        os.makedirs(os.path.join(data_cache,class_id,'test'))

    files_list = [x for x in os.listdir(os.path.join(root_dir,class_ids[0])) if x[-4:]=='.wav']
    num_files = len(files_list)

    indices = np.arange(0,num_files)
    np.random.seed(7)
    np.random.shuffle(indices)

    train_split = math.floor(train_percent/100*num_files)
    test_split = math.floor(test_percent/100*num_files)

    for phase,(start, end) in zip(['train','test'],[(0,train_split),(num_files-test_split,num_files)]):
        
        total_duration=0
        total_clips=0
        
        if use_genders!='None':
            male_duration = 0
            female_duration = 0
            male_clips = 0
            female_clips = 0


        for i in range(start,end):
            fileA, fileB, file=get_filenames(files_list[indices[i]])
            if librosa.get_duration(filename=os.path.join(root_dir,class_ids[0],fileA)) < 1: #Skipping very short files
                continue
            if use_genders!='None' and phase!='test':
                if annotations[file] not in use_genders:
                    continue
            shutil.copyfile(os.path.join(root_dir,class_ids[0],fileA),os.path.join(data_cache,class_ids[0],phase,file))
            shutil.copyfile(os.path.join(root_dir,class_ids[1],fileB),os.path.join(data_cache,class_ids[1],phase,file))
            duration=librosa.get_duration(filename=os.path.join(data_cache,class_ids[0],phase,file))
            
            total_duration+=duration
            total_clips+=1

            if use_genders!='None':
                if annotations[file] == 'M':
                    male_clips+=1
                    male_duration+=duration
                else:
                    female_clips+=1
                    female_duration+=duration

        print(f'{total_duration} seconds ({total_clips} clips) of Audio saved to {phase}.')
        if use_genders!='None':
            print(f'{male_duration} seconds ({male_clips} clips) of male Audio in {phase}.')
            print(f'{female_duration} seconds ({female_clips} clips) of female Audio in {phase}.')

def fetch_from_npy(train_path,test_path,data_cache,sr):

    """
    Fetch train and test sets saved as npy and save them as audio files in data_cache dir.
    Created by Leander Maben.
    """

    train_set = np.load(train_path)
    test_set = np.load(test_path)

    os.makedirs(os.path.join(data_cache,'clean','train'))
    os.makedirs(os.path.join(data_cache,'clean','test'))
    os.makedirs(os.path.join(data_cache,'noisy','train'))
    os.makedirs(os.path.join(data_cache,'noisy','test'))

    for i in range(train_set.shape[0]):
        sf.write(os.path.join(data_cache,'clean','train',f'{i}_audio.wav'),train_set[i,:,0],sr)
        sf.write(os.path.join(data_cache,'noisy','train',f'{i}_audio.wav'),train_set[i,:,1],sr)

    for i in range(test_set.shape[0]):
        sf.write(os.path.join(data_cache,'clean','test',f'{i}_audio.wav'),test_set[i,:,0],sr)
        sf.write(os.path.join(data_cache,'noisy','test',f'{i}_audio.wav'),test_set[i,:,1],sr)

def fetch_with_codec(clean_path,codec,data_cache,train_speakers,test_speakers,train_duration_max,test_duration_max):
    """
    Transfer audio files to a convinient location for processing with train,test split.
    Generate the noisy data set from clean dataset using the specified codec.
    Arguments:
    clean_path(str) - Root directory where files of specified classes are present in subdirectories.
    codec(str) - Name of codec to be used.
    data_cache(str) - Root directory to store data. Data is stored in clean and noisy sub-directories.
    Created By Leander Maben. 
    """
    if codec == 'g726':
        print('Using codec g726 with bit rate 16k')
    elif codec == 'ogg':
        print('Using codec ogg with bit rate 4.5k')
    elif codec == 'g723_1':
        print('Using codec g723_1 with bit rate 6.3k')
    elif codec == 'gsm':
        print('Using codec gsm with bit rate 13k')
    elif codec == 'codec2':
        print(f'Using codec codec2 with bit rate {defaults["codec2_bitrate"]}')

    train_clips = []
    test_clips = []
    
    for file in os.listdir(clean_path):
        if file[:16] in train_speakers:
            train_clips.append(file)
        elif file[:16] in test_speakers:
            test_clips.append(file)

    np.random.seed(7)
    np.random.shuffle(train_clips)
    np.random.seed(8)
    np.random.shuffle(test_clips)


    for class_id in ['clean','noisy']:
        os.makedirs(os.path.join(data_cache,class_id,'train'))
        os.makedirs(os.path.join(data_cache,class_id,'test'))

    duration_saved = {'train':0, 'test':0}
    total_clips = {'train':0, 'test':0}
    duration_max = {'train':train_duration_max, 'test':test_duration_max}
    clips = train_clips+test_clips
    phase_labels = ['train']*len(train_clips)+['test']*len(test_clips)

    for file, phase in zip(clips,phase_labels):
        if librosa.get_duration(filename=os.path.join(clean_path,file)) + duration_saved[phase] < duration_max[phase] and librosa.get_duration(filename=os.path.join(clean_path,file))>1:
            shutil.copyfile(os.path.join(clean_path,file),os.path.join(data_cache,'clean',phase,file))
            clean_data, clean_sr = librosa.load(os.path.join(clean_path,file), sr=None)
            
            if codec == 'g726':
                file_orig = os.path.join(data_cache,'clean',phase,file)
                file_8k = os.path.join(data_cache,'noisy',phase,file[:-4]+'_8k.wav')
                file_codec = os.path.join(data_cache,'noisy',phase,file[:-4]+'_fmt.wav')
                file_out = os.path.join(data_cache,'noisy',phase,file)
                run(f'ffmpeg -hide_banner -loglevel error -i {file_orig} -ar 8k -y {file_8k}')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_8k} -acodec g726 -b:a 16k {file_codec}')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_codec} -ar 8k -y {file_out}')
                os.remove(file_8k)
                os.remove(file_codec)
            elif codec == 'ogg':
                file_orig = os.path.join(data_cache,'clean',phase,file)
                file_codec = os.path.join(data_cache,'noisy',phase,file[:-4]+'_fmt.ogg')
                file_out = os.path.join(data_cache,'noisy',phase,file)
                run(f'ffmpeg -hide_banner -loglevel error -i {file_orig} -c:a libopus -b:a 4.5k -ar 8000 {file_codec}')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_codec} -ar 8000 {file_out}')
                os.remove(file_codec)
            elif codec == 'g723_1':
                file_orig = os.path.join(data_cache,'clean',phase,file)
                file_codec = os.path.join(data_cache,'noisy',phase,file[:-4]+'_fmt.wav')
                file_out = os.path.join(data_cache,'noisy',phase,file)
                run(f'ffmpeg -hide_banner -loglevel error -i {file_orig} -acodec g723_1 -b:a 6.3k -ar 8000 {file_codec}')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_codec} -ar 8000 {file_out}')
                os.remove(file_codec)
            elif codec == 'gsm':
                file_orig = os.path.join(data_cache,'clean',phase,file)
                file_codec = os.path.join(data_cache,'noisy',phase,file[:-4]+'_fmt.gsm')
                file_out = os.path.join(data_cache,'noisy',phase,file)
                run(f'ffmpeg -hide_banner -loglevel error -i {file_orig} -acodec libgsm -b:a 13k -ar 8000 {file_codec}')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_codec} -ar 8000 {file_out}')
                os.remove(file_codec)
            elif codec == 'codec2':
                if not os.path.exists('codec2'):
                    ## Cloning codec2 repo and building
                    cwd = os.getcwd()
                    run('git clone https://github.com/drowe67/codec2.git')
                    os.chdir('codec2')
                    os.mkdir('build_linux')
                    os.chdir('build_linux')
                    run('cmake ..')
                    run('make')
                    os.chdir(cwd)
                file_orig = os.path.join(data_cache,'clean',phase,file)
                file_8k = os.path.join(data_cache,'noisy',phase,file[:-4]+'_8k.wav')
                file_raw_input = os.path.join(data_cache,'noisy',phase,file[:-4]+'_in.raw')
                file_enc = os.path.join(data_cache,'noisy',phase,file[:-4]+'_enc.bit')
                file_raw_out = os.path.join(data_cache,'noisy',phase,file[:-4]+'_out.raw')
                file_out = os.path.join(data_cache,'noisy',phase,file)
                run(f'ffmpeg -hide_banner -loglevel error -i {file_orig} -ar 8k -y {file_8k}')
                run(f'ffmpeg -i {file_8k} -f s16le -acodec pcm_s16le {file_raw_input}') #Convert to raw
                run(f'codec2/build_linux/src/c2enc {defaults["codec2_bitrate"]} {file_raw_input} {file_enc}') # Encode
                run(f'codec2/build_linux/src/c2dec {defaults["codec2_bitrate"]} {file_enc} {file_raw_out}') #Decode
                run(f'ffmpeg -f s16le -ar 8k -ac 1 -i {file_raw_out} {file_out}') #Convert to wav
                
                os.remove(file_8k)
                os.remove(file_raw_input)
                os.remove(file_enc)
                os.remove(file_raw_out)

            duration_saved[phase]+=librosa.get_duration(filename=os.path.join(data_cache,'clean',phase,file))
            total_clips[phase]+=1
            

    print(f"Saved {duration_saved['train']} seconds and {total_clips['train']} clips of audio to train.")
    print(f"Saved {duration_saved['test']} seconds and {total_clips['test']} clips of audio to test.")
    
       

def additive_noise(clean_path,noise_file,data_cache,train_speakers,test_speakers,train_duration_max,test_duration_max):
    noise, noise_sr = librosa.load(noise_file, sr=None)
    # noise_train = noise[0:noise.shape[0]//2]
    # noise_test = noise[noise.shape[0]//2:]

    noise_train = noise
    noise_test = noise

    train_clips = []
    test_clips = []
    
    for file in os.listdir(clean_path):
        if file[:16] in train_speakers:
            train_clips.append(file)
        elif file[:16] in test_speakers:
            test_clips.append(file)

    np.random.seed(7)
    np.random.shuffle(train_clips)
    np.random.seed(8)
    np.random.shuffle(test_clips)

    os.makedirs(os.path.join(data_cache,'clean','train'))
    os.makedirs(os.path.join(data_cache,'clean','test'))
    os.makedirs(os.path.join(data_cache,'noisy','train'))
    os.makedirs(os.path.join(data_cache,'noisy','test'))

    train_duration_saved = 0
    test_duration_saved = 0
    for clip in train_clips:
        if librosa.get_duration(filename=os.path.join(clean_path,clip)) + train_duration_saved < train_duration_max and librosa.get_duration(filename=os.path.join(clean_path,clip))>1:
            shutil.copyfile(os.path.join(clean_path,clip),os.path.join(data_cache,'clean','train',clip))
            clean_data, clean_sr = librosa.load(os.path.join(clean_path,clip), sr=None)
            assert clean_sr == noise_sr
            assert noise_train.shape[0]>clean_data.shape[0]
            start = np.random.randint(0,noise_train.shape[0]-clean_data.shape[0]+1)
            result = defaults["clean_additive_weight"]*clean_data + defaults["noisy_additive_weight"]*noise_train[start:start+clean_data.shape[0]]
            sf.write(os.path.join(data_cache,'noisy','train',clip),result,clean_sr)
            train_duration_saved+=librosa.get_duration(filename=os.path.join(clean_path,clip))

    print(f'Saved {train_duration_saved} seconds of audio to train.')
    
    for clip in test_clips:
        if librosa.get_duration(filename=os.path.join(clean_path,clip)) + test_duration_saved < test_duration_max and librosa.get_duration(filename=os.path.join(clean_path,clip))>1:
            shutil.copyfile(os.path.join(clean_path,clip),os.path.join(data_cache,'clean','test',clip))
            clean_data, clean_sr = librosa.load(os.path.join(clean_path,clip), sr=None)
            assert clean_sr == noise_sr
            assert noise_test.shape[0]>clean_data.shape[0]
            start = np.random.randint(0,noise_test.shape[0]-clean_data.shape[0]+1)
            result = 0.5*clean_data + 0.5*noise_test[start:start+clean_data.shape[0]]
            sf.write(os.path.join(data_cache,'noisy','test',clip),result,clean_sr)
            test_duration_saved+=librosa.get_duration(filename=os.path.join(clean_path,clip))

    print(f'Saved {test_duration_saved} seconds of audio to test.')

def rats_noise(root_dir,data_cache,train_speakers,test_speakers,train_duration_max,test_duration_max):

    train_clips = []
    test_clips = []
    
    clean_path = os.path.join(root_dir,'clean')
    noisy_path = os.path.join(root_dir,'noisy')

    for file in os.listdir(clean_path):
        if file[:16] in train_speakers:
            train_clips.append(file)
        elif file[:16] in test_speakers:
            test_clips.append(file)

    np.random.seed(7)
    np.random.shuffle(train_clips)
    np.random.seed(8)
    np.random.shuffle(test_clips)

    os.makedirs(os.path.join(data_cache,'clean','train'))
    os.makedirs(os.path.join(data_cache,'clean','test'))
    os.makedirs(os.path.join(data_cache,'noisy','train'))
    os.makedirs(os.path.join(data_cache,'noisy','test'))

    train_duration_saved = 0
    test_duration_saved = 0
    for clip in train_clips:
        clean_clip,noisy_clip,renamed_clip = get_filenames(clip)
        if librosa.get_duration(filename=os.path.join(root_dir,'clean',clip)) + train_duration_saved < train_duration_max and librosa.get_duration(filename=os.path.join(root_dir,'clean',clip))>1:
            shutil.copyfile(os.path.join(clean_path,clip),os.path.join(data_cache,'clean','train',renamed_clip))
            shutil.copyfile(os.path.join(noisy_path,noisy_clip),os.path.join(data_cache,'noisy','train',renamed_clip))
            train_duration_saved+=librosa.get_duration(filename=os.path.join(clean_path,clip))

    print(f'Saved {train_duration_saved} seconds of audio to train.')
    
    for clip in test_clips:
        clean_clip,noisy_clip,renamed_clip = get_filenames(clip)
        if librosa.get_duration(filename=os.path.join(clean_path,clip)) + test_duration_saved < test_duration_max and librosa.get_duration(filename=os.path.join(clean_path,clip))>1:
            shutil.copyfile(os.path.join(clean_path,clip),os.path.join(data_cache,'clean','test',renamed_clip))
            shutil.copyfile(os.path.join(noisy_path,noisy_clip),os.path.join(data_cache,'noisy','test',renamed_clip))
            test_duration_saved+=librosa.get_duration(filename=os.path.join(clean_path,clip))

    print(f'Saved {test_duration_saved} seconds of audio to test.')

def transfer_timit(timit_dir,data_cache,val_speakers,test_speakers,train_duration_max,val_duaration_max,test_duration_max,noise_type,noise_db, val_required = False):


    clean_train_path = os.path.join(timit_dir,'train','clean')
    noisy_path = os.path.join(timit_dir,'test','noisy',noise_type,f'{noise_db}dB')

    print(f'Clean (Train) Path: {clean_train_path}')
    print(f'Noisy Path: {noisy_path}')

    noisy_speakers = os.listdir(noisy_path)

    train_speakers_noisy = list(set(noisy_speakers)-set(test_speakers)-set(val_speakers))
    
    print(f'Train Speakers (Noisy): {train_speakers_noisy}')
    print(f'Validation Speakers (Noisy): {val_speakers}')
    print(f'Test Speakers (Noisy): {test_speakers}')
    
    clean_clips_train_all = glob.glob(os.path.join(clean_train_path,'**/*.wav'), recursive=True)
    noisy_clips_train_all = []
    for speaker in train_speakers_noisy:
        noisy_clips_train_all.extend(glob.glob(os.path.join(noisy_path,speaker,'*.wav')))
    
    val_clips_all = []
    for speaker in val_speakers:
        for clip in os.listdir(os.path.join(noisy_path,speaker)):
            val_clips_all.append(f'{speaker}/{clip}') #Saving only speaker/clip.wav
    
    test_clips_all = []
    for speaker in test_speakers:
        for clip in os.listdir(os.path.join(noisy_path,speaker)):
            test_clips_all.append(f'{speaker}/{clip}') #Saving only speaker/clip.wav

    np.random.seed(7)
    np.random.shuffle(clean_clips_train_all)
    np.random.seed(8)
    np.random.shuffle(noisy_clips_train_all)
    np.random.seed(9)
    np.random.shuffle(val_clips_all)
    np.random.seed(10)
    np.random.shuffle(test_clips_all)

    os.makedirs(os.path.join(data_cache,'clean','train'))
    os.makedirs(os.path.join(data_cache,'clean','test'))
    os.makedirs(os.path.join(data_cache,'noisy','train'))
    os.makedirs(os.path.join(data_cache,'noisy','test'))
    if val_required:
        os.makedirs(os.path.join(data_cache,'clean','val'))
        os.makedirs(os.path.join(data_cache,'noisy','val'))


    clean_train_duration_saved = 0
    for clip in clean_clips_train_all:
        basename = os.path.basename(clip)
        if librosa.get_duration(filename=clip) + clean_train_duration_saved < train_duration_max and librosa.get_duration(filename=clip)>1:
            shutil.copyfile(clip,os.path.join(data_cache,'clean','train',basename))
            clean_train_duration_saved+=librosa.get_duration(filename=clip)

    print(f'Saved {clean_train_duration_saved} seconds of clean audio to train.')

    noisy_train_duration_saved = 0
    for clip in noisy_clips_train_all:
        basename = os.path.basename(clip)
        if librosa.get_duration(filename=clip) + noisy_train_duration_saved < train_duration_max and librosa.get_duration(filename=clip)>1:
            shutil.copyfile(clip,os.path.join(data_cache,'noisy','train',basename))
            noisy_train_duration_saved+=librosa.get_duration(filename=clip)

    print(f'Saved {noisy_train_duration_saved} seconds of noisy audio to train.')
    
    val_duration_saved = 0
    for clip in val_clips_all:
        basename = os.path.basename(clip)
        if librosa.get_duration(filename=os.path.join(timit_dir,'test','clean',clip)) + val_duration_saved < val_duration_max and librosa.get_duration(filename=os.path.join(timit_dir,'test','clean',clip))>1:
            shutil.copyfile(os.path.join(timit_dir,'test','clean',clip),os.path.join(data_cache,'clean','val',basename))
            shutil.copyfile(os.path.join(noisy_path,clip),os.path.join(data_cache,'noisy','val',basename))
            val_duration_saved+=librosa.get_duration(filename=os.path.join(timit_dir,'test','clean',clip))

    print(f'Saved {val_duration_saved} seconds of audio to val.')

    test_duration_saved = 0
    for clip in test_clips_all:
        basename = os.path.basename(clip)
        if librosa.get_duration(filename=os.path.join(timit_dir,'test','clean',clip)) + test_duration_saved < test_duration_max and librosa.get_duration(filename=os.path.join(timit_dir,'test','clean',clip))>1:
            shutil.copyfile(os.path.join(timit_dir,'test','clean',clip),os.path.join(data_cache,'clean','test',basename))
            shutil.copyfile(os.path.join(noisy_path,clip),os.path.join(data_cache,'noisy','test',basename))
            test_duration_saved+=librosa.get_duration(filename=os.path.join(timit_dir,'test','clean',clip))

    print(f'Saved {test_duration_saved} seconds of audio to test.')

def transfer_timit_parallel(timit_dir,data_cache,test_speakers,train_duration_max,test_duration_max,noise_type,noise_db):
    
    clean_train_path = os.path.join(timit_dir,'test','clean')
    noisy_path = os.path.join(timit_dir,'test','noisy',noise_type,f'{noise_db}dB')

    print(f'Clean (Train) Path: {clean_train_path}')
    print(f'Noisy Path: {noisy_path}')

    speakers = os.listdir(noisy_path)

    train_speakers = list(set(speakers)-set(test_speakers))
    
    print(f'Train Speakers (Noisy): {train_speakers}')
    print(f'Test Speakers (Noisy): {test_speakers}')
    
    train_clips_all = []
    test_clips_all = []
    for speaker in train_speakers:
        for clip in os.listdir(os.path.join(noisy_path,speaker)):
            train_clips_all.append(f'{speaker}/{clip}') #Saving only speaker/clip.wav
    for speaker in test_speakers:
        for clip in os.listdir(os.path.join(noisy_path,speaker)):
            test_clips_all.append(f'{speaker}/{clip}') #Saving only speaker/clip.wav

    np.random.seed(7)
    np.random.shuffle(train_clips_all)
    np.random.seed(9)
    np.random.shuffle(test_clips_all)

    os.makedirs(os.path.join(data_cache,'clean','train'))
    os.makedirs(os.path.join(data_cache,'clean','test'))
    os.makedirs(os.path.join(data_cache,'noisy','train'))
    os.makedirs(os.path.join(data_cache,'noisy','test'))

    train_duration_saved = 0
    for clip in train_clips_all:
        basename = os.path.basename(clip)
        if librosa.get_duration(filename=os.path.join(timit_dir,'test','clean',clip)) + train_duration_saved < train_duration_max and librosa.get_duration(filename=os.path.join(timit_dir,'test','clean',clip))>1:
            shutil.copyfile(os.path.join(timit_dir,'test','clean',clip),os.path.join(data_cache,'clean','train',basename))
            shutil.copyfile(os.path.join(noisy_path,clip),os.path.join(data_cache,'noisy','train',basename))
            train_duration_saved+=librosa.get_duration(filename=os.path.join(timit_dir,'test','clean',clip))

    print(f'Saved {train_duration_saved} seconds of audio to train.')

    test_duration_saved = 0
    for clip in test_clips_all:
        basename = os.path.basename(clip)
        if librosa.get_duration(filename=os.path.join(timit_dir,'test','clean',clip)) + test_duration_saved < test_duration_max and librosa.get_duration(filename=os.path.join(timit_dir,'test','clean',clip))>1:
            shutil.copyfile(os.path.join(timit_dir,'test','clean',clip),os.path.join(data_cache,'clean','test',basename))
            shutil.copyfile(os.path.join(noisy_path,clip),os.path.join(data_cache,'noisy','test',basename))
            test_duration_saved+=librosa.get_duration(filename=os.path.join(timit_dir,'test','clean',clip))

    print(f'Saved {test_duration_saved} seconds of audio to test.')






if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare Data')
    parser.add_argument('--audio_data_path', dest = 'audio_path', type=str, default=defaults['audio_data_path'], help="Path to audio root folder")
    parser.add_argument('--source_sub_directories', dest = 'sub_directories',type=str, default=defaults["source_sub_directories"], help="Sub directories for data")
    parser.add_argument('--data_cache', dest='data_cache', type=str, default=defaults["data_cache"], help="Directory to Store data and meta data.")
    parser.add_argument('--annotations_path', dest='annotations_path', type=str, default=defaults["annotations"], help='Path to CSV containing gender annotations.Use only if --use_genders is not None.Ignored for --transfer_mode [spectrogram|npy]')
    parser.add_argument('--train_percent', dest='train_percent', type=int, default=defaults["train_percent"], help="Percentage for train split.Ignored for --transfer_mode npy or additive_noise.")
    parser.add_argument('--test_percent', dest='test_percent', type=int, default=defaults["test_percent"], help="Percentage for test split.Ignored for --transfer_mode npy or additive_noise.")
    parser.add_argument('--size_multiple', dest='size_multiple', type=int, default=defaults["size_multiple"], help="Required Factor of Dimensions ONLY if spectrogram mode of tranfer is used")
    parser.add_argument('--sampling_rate', dest='sampling_rate', type=int, default=defaults["sampling_rate"], help="Sampling rate for audio. Use if tranfer_mode is spectrogram or npy")
    parser.add_argument('--transfer_mode', dest='transfer_mode', type=str, choices=['rats','spectrogram','npy','codec','additive_noise','timit', 'timit_parallel'], default=defaults["transfer_mode"], help='Transfer files as raw audio ,converted spectrogram, from npy files, using codecor adding noise.')
    parser.add_argument('--use_genders', dest='use_genders', type=str, default=defaults["use_genders"], help='Genders to include in train set. Pass None if you do not want to check genders.Ignored for --transfer_mode [spectrogram|npy]')
    parser.add_argument('--npy_train_source', dest='npy_train_source', type=str, default=defaults["npy_train"], help='Path where npy train set is present.')
    parser.add_argument('--npy_test_source', dest='npy_test_source', type=str, default=defaults["npy_test"], help='Path where npy test set is present.')
    parser.add_argument('--clean_path', dest='clean_path', type=str, default=defaults["clean_path"], help='Path to clean audio files. Only use if --transfer_mode is codec or additive_noise.')
    parser.add_argument('--codec_name', dest='codec_name', type=str, default=defaults["codec_name"], choices=['g726','ogg', 'g723_1','gsm','codec2'], help='Name of codec to be used. Only use if --transfer_mode is codec.')
    parser.add_argument('--noise_file', dest='noise_file', type=str, default=defaults["noise_file"], help="Path to file containing noise.")
    parser.add_argument('--train_speakers', dest='train_speakers', nargs='+' ,type=str, default=defaults["train_speakers"], help="Ids of speakers to be used in train set. Use only if --transfer_mode is additive_noise.")
    parser.add_argument('--test_speakers', dest='test_speakers', nargs='+' ,type=str, default=defaults["test_speakers"], help="Ids of speakers to be used in test set. Use only if --transfer_mode is additive_noise.")
    parser.add_argument('--train_duration_max', dest='train_duration_max' ,type=int, default=defaults["train_duration_max"], help="Max duration of train dataset. Use only if --transfer_mode is additive_noise.")
    parser.add_argument('--val_duration_max', dest='val_duration_max' ,type=int, default=defaults["val_duration_max"], help="Max duration of val dataset. Use only if --transfer_mode is additive_noise.")
    parser.add_argument('--test_duration_max', dest='test_duration_max' ,type=int, default=defaults["test_duration_max"], help="Max duration of test dataset. Use only if --transfer_mode is additive_noise.")
    parser.add_argument('--noise_dB', dest='noise_dB' ,type=int, default=defaults["timit_noise_dB"], help="SNR for TIMIT noise dataset. Use only if --transfer_mode is timit.")
    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] = ' % arg, getattr(args, arg))
    if args.transfer_mode == 'spectrogram':
        for class_id in args.sub_directories:        
            preprocess_dataset_spectrogram(os.path.join(args.audio_path,class_id),class_id,args.sampling_rate, args.data_cache, args.train_percent, args.test_percent, args.size_multiple)
    elif args.transfer_mode == 'rats':
        rats_noise(args.audio_path, args.data_cache, args.train_speakers, args.test_speakers, args.train_duration_max, args.test_duration_max)
    elif args.transfer_mode == 'npy':
        fetch_from_npy(args.npy_train_source, args.npy_test_source,args.data_cache, args.sampling_rate)
    elif args.transfer_mode == 'codec':
        fetch_with_codec(args.clean_path, args.codec_name, args.data_cache, args.train_speakers, args.test_speakers, args.train_duration_max, args.test_duration_max)
    elif args.transfer_mode == 'additive_noise':
        additive_noise(args.clean_path, args.noise_file, args.data_cache, args.train_speakers, args.test_speakers, args.train_duration_max, args.test_duration_max)
    elif args.transfer_mode == 'timit':
        transfer_timit(defaults["timit_dir"],args.data_cache, defaults["timit_val_spakers"],defaults["timit_test_speakers"], args.train_duration_max, args.val_duration_max, args.test_duration_max, defaults["timit_noise_type"],args.noise_dB, defaults['val_required'])
    elif args.transfer_mode == 'timit_parallel':
        transfer_timit_parallel(defaults["timit_dir"],args.data_cache, defaults["timit_test_speakers"], args.train_duration_max, args.test_duration_max, defaults["timit_noise_type"],args.noise_dB)
    