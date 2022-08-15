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
from scipy import signal



def run(command):
    #print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)

"""
Code to transfer audio data from a source folder to a target folder with train and test splits.
Change AUDIO_DATA_PATH_DEFAULT to point to root dir such that (or use command line argument)
-root
    -subdirectory[0]
        -sample_0
        -....
    -subdirectory[1]
        -sample_0
        -sample_1
Change CACHE_DEFAULT to the directory where you want data to be stored.
There are 2 options to transfer data -> It can be transferred as audio files or as spectrograms.
Spectrograms are generated using MelGAN.However, MelGAN performs poorly when converting noisy spectrogram back to audio.
Hence it is recommended to use the 'audio' option for the argument --transfer_mode (It is already set as default)
"""

AUDIO_DATA_PATH_DEFAULT = '/content/drive/MyDrive/NTU - Speech Augmentation/Parallel_speech_data'
SUBDIRECTORIES_DEFAULT = ['clean','noisy']
CACHE_DEFAULT = '/content/Pix2Pix-VC/data_cache'
SAMPLING_RATE = 8000
CSV_PATH_DEFAULT = '/content/drive/MyDrive/NTU - Speech Augmentation/annotations.csv' #Only if --use_genders is not None.Ignored for --transfer_mode [spectrogram|npy]
NPY_TRAIN_DEFAULT = '/content/drive/MyDrive/NTU - Speech Augmentation/rats_train.npy' #Only if --transfer_mode is npy
NPY_TEST_DEFAULT = '/content/drive/MyDrive/NTU - Speech Augmentation/rats_valid.npy' #Only if --transfer_mode is npy

## mel function inspired from https://github.com/GANtastic3/MaskCycleGAN-VC

def mel(wavspath):
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
        wav_orig, _ = librosa.load(wavpath, sr=SAMPLING_RATE, mono=True)
        spec = vocoder(torch.tensor([wav_orig]))
        #print(f'Spectrogram shape: {spec.shape}')
        if spec.shape[-1] >= 64:    # training sample consists of 64 frames
            mel_list.append(spec.cpu().detach().numpy()[0])
            info['duration']+=librosa.get_duration(filename=wavpath)
            info['records_count']+=1   
            filenames.append(file)       
    return mel_list, filenames, info





def preprocess_dataset_spectrogram(data_path, class_id, args):
    """Preprocesses dataset of .wav files by converting to Mel-spectrograms.
    Args:
        data_path (str): Directory containing .wav files of the speaker.
        speaker_id (str): ID of the speaker.
        cache_folder (str, optional): Directory to hold preprocessed data. Defaults to './cache/'.
        Modified By Leander Maben.
    """

    print(f"Preprocessing data for class: {class_id}.")

    cache_folder = args.data_cache

    mel_list, filenames, info = mel(data_path)

    if not os.path.exists(os.path.join(cache_folder, class_id)):
        os.makedirs(os.path.join(cache_folder, class_id, 'meta'))
        os.makedirs(os.path.join(cache_folder, class_id, 'train'))
        os.makedirs(os.path.join(cache_folder, class_id, 'val'))
        os.makedirs(os.path.join(cache_folder, class_id, 'test'))
    
    indices = np.arange(0,len(mel_list))
    np.random.seed(0)
    np.random.shuffle(indices)

    train_split = math.floor(args.train_percent/100*len(mel_list))
    val_split = math.floor(args.val_percent/100*len(mel_list))
    
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
            top_pad = args.size_multiple - freq_len % args.size_multiple if freq_len % args.size_multiple!=0 else 0
            right_pad = args.size_multiple - time_len % args.size_multiple if time_len % args.size_multiple!=0 else 0
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

def AddNoiseFloor(data):
    frameSz = 128
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

def time_align(data1, data2, sr):
    nfft = 256
    hop_length = 1  # hop_length = win_length or frameSz - overlapSz
    win_length = 256

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


    delay = delay_arr[np.argmax(corr)]

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

    return data1, data2


def get_filenames(fileNameA):
    """
    Custom function for this specific dataset.
    It returns the names of corresponding files in the 2 classes along with the common name by which it should be saved.
    Args:
    fileNameA(str) : Filename in the first class
    Created By Leander Maben
    """

    return fileNameA, fileNameA[:32]+'-A.wav', fileNameA[:32]+'.wav'



def transfer_aligned_audio_raw(root_dir,class_ids,data_cache,train_percent,test_percent, use_genders, annotations_path):
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
        print(f'{male_duration} seconds ({male_clips} clips) of male Audio in {phase}.')
        print(f'{female_duration} seconds ({female_clips} clips) of female Audio in {phase}.')

def fetch_from_npy(train_path,test_path,data_cache, sr=SAMPLING_RATE):

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

def fetch_with_codec(clean_path,codec,data_cache,train_percent,test_percent, use_genders, annotations_path):
    """
    Transfer audio files to a convinient location for processing with train,test,validation split.
    Important Note: The splitting of data by percent is based on file numbers and not on cummulative duration
    of clips. Moreover, it does not take into the account the number of clips that are discarded for being less than 1 second long.
    Arguments:
    clean_path(str) - Root directory where files of specified classes are present in subdirectories.
    codec(str) - Name of codec to be used.
    data_cache(str) - Root directory to store data. Data is stored in clean and noisy sub-directories.
    train_percent(int) - Percent of data clips in train split
    test_percent(int) - Percent of data clips in test split
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

    if use_genders != 'None':
        annotations = {}
        anno_csv = pd.read_csv(annotations_path)
        for i in range(len(anno_csv)):
            row=anno_csv.iloc[i]
            annotations[row['file']]=row['gender']


    for class_id in ['clean','noisy']:
        os.makedirs(os.path.join(data_cache,class_id,'train'))
        os.makedirs(os.path.join(data_cache,class_id,'test'))

    files_list = [x for x in os.listdir(clean_path) if x[-4:]=='.wav']
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
            fileA, _, file=get_filenames(files_list[indices[i]])
            if librosa.get_duration(filename=os.path.join(clean_path,fileA)) < 1: #Skipping very short files
                continue
            if use_genders!='None' and phase!='test':
                if annotations[file] not in use_genders:
                    continue
            
            shutil.copyfile(os.path.join(clean_path,fileA),os.path.join(data_cache,'clean',phase,file))

            if codec == 'g726':
                file_orig = os.path.join(data_cache,'clean',phase,file)
                file_8k = os.path.join(data_cache,'noisy',phase,file[:-4]+'_8k.wav')
                file_codec = os.path.join(data_cache,'noisy',phase,file[:-4]+'_fmt.wav')
                file_temp = os.path.join(data_cache,'noisy',phase,file[:-4]+'_temp.wav')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_orig} -ar 8k -y {file_8k}')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_8k} -acodec g726 -b:a 16k {file_codec}')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_codec} -ar 8k -y {file_temp}')
                os.remove(file_8k)
                os.remove(file_codec)
            elif codec == 'ogg':
                file_orig = os.path.join(data_cache,'clean',phase,file)
                file_codec = os.path.join(data_cache,'noisy',phase,file[:-4]+'_fmt.ogg')
                file_temp = os.path.join(data_cache,'noisy',phase,file[:-4]+'_temp.wav')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_orig} -c:a libopus -b:a 4.5k -ar 8000 {file_codec}')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_codec} -ar 8000 {file_temp}')
                os.remove(file_codec)
            elif codec == 'g723_1':
                file_orig = os.path.join(data_cache,'clean',phase,file)
                file_codec = os.path.join(data_cache,'noisy',phase,file[:-4]+'_fmt.wav')
                file_temp = os.path.join(data_cache,'noisy',phase,file[:-4]+'_temp.wav')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_orig} -acodec g723_1 -b:a 6.3k -ar 8000 {file_codec}')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_codec} -ar 8000 {file_temp}')
                os.remove(file_codec)
            elif codec == 'gsm':
                file_orig = os.path.join(data_cache,'clean',phase,file)
                file_codec = os.path.join(data_cache,'noisy',phase,file[:-4]+'_fmt.gsm')
                file_temp = os.path.join(data_cache,'noisy',phase,file[:-4]+'_temp.wav')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_orig} -acodec libgsm -b:a 13k -ar 8000 {file_codec}')
                run(f'ffmpeg -hide_banner -loglevel error -i {file_codec} -ar 8000 {file_temp}')
                os.remove(file_codec)

            os.remove(file_orig)
            data1, _ = librosa.load(file_temp,sr=SAMPLING_RATE)
            data2, _ = librosa.load(os.path.join(clean_path,fileA),sr=SAMPLING_RATE)
            data1, data2 = time_align(data1,data2,sr=SAMPLING_RATE)

            sf.write(os.path.join(data_cache,'noisy',phase,file),data1,SAMPLING_RATE)
            sf.write(os.path.join(data_cache,'clean',phase,file),data2,SAMPLING_RATE)
            os.remove(file_temp)
            
            assert librosa.get_duration(filename=os.path.join(data_cache,'clean',phase,file)) == librosa.get_duration(filename=os.path.join(data_cache,'noisy',phase,file))
            duration=librosa.get_duration(filename=os.path.join(data_cache,'clean',phase,file))
            
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
        print(f'{male_duration} seconds ({male_clips} clips) of male Audio in {phase}.')
        print(f'{female_duration} seconds ({female_clips} clips) of female Audio in {phase}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare Data')
    parser.add_argument('--audio_data_path', dest = 'audio_path', type=str, default=AUDIO_DATA_PATH_DEFAULT, help="Path to audio root folder")
    parser.add_argument('--source_sub_directories', dest = 'sub_directories',type=str, default=SUBDIRECTORIES_DEFAULT, help="Sub directories for data")
    parser.add_argument('--data_cache', dest='data_cache', type=str, default=CACHE_DEFAULT, help="Directory to Store data and meta data.")
    parser.add_argument('--annotations_path', dest='annotations_path', type=str, default=CSV_PATH_DEFAULT, help='Path to CSV containing gender annotations.Use only if --use_genders is not None.Ignored for --transfer_mode [spectrogram|npy]')
    parser.add_argument('--train_percent', dest='train_percent', type=int, default=10, help="Percentage for train split.Ignored for --transfer_mode npy.")
    parser.add_argument('--test_percent', dest='test_percent', type=int, default=15, help="Percentage for test split.Ignored for --transfer_mode npy.")
    parser.add_argument('--size_multiple', dest='size_multiple', type=int, default=4, help="Required Factor of Dimensions ONLY if spectrogram mode of tranfer is used")
    parser.add_argument('--transfer_mode', dest='transfer_mode', type=str, choices=['audio','spectrogram','npy','codec'], default='audio', help='Transfer files as raw audio ,converted spectrogram, from npy files or using codec.')
    parser.add_argument('--use_genders', dest='use_genders', type=str, default=['M','F'], help='Genders to include in train set. Pass None if you do not want to check genders.Ignored for --transfer_mode [spectrogram|npy]')
    parser.add_argument('--npy_train_source', dest='npy_train_source', type=str, default=NPY_TRAIN_DEFAULT, help='Path where npy train set is present.')
    parser.add_argument('--npy_test_source', dest='npy_test_source', type=str, default=NPY_TEST_DEFAULT, help='Path where npy test set is present.')
    parser.add_argument('--codec_clean_path', dest='codec_clean_path', type=str, default=os.path.join(AUDIO_DATA_PATH_DEFAULT,'clean'), help='Path to clean audio files. Only use if --transfer_mode is codec.')
    parser.add_argument('--codec_name', dest='codec_name', type=str, default='g726', choices=['g726','ogg', 'g723_1','gsm'], help='Name of codec to be used. Only use if --transfer_mode is codec.')
    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] = ' % arg, getattr(args, arg))
    if args.transfer_mode == 'spectrogram':
        for class_id in args.sub_directories:        
            preprocess_dataset_spectrogram(os.path.join(args.audio_path,class_id),class_id,args)
    elif args.transfer_mode == 'audio':
        transfer_aligned_audio_raw(args.audio_path,args.sub_directories,args.data_cache,args.train_percent,args.test_percent, args.use_genders, args.annotations_path)
    elif args.transfer_mode == 'npy':
        fetch_from_npy(args.npy_train_source, args.npy_test_source,args.data_cache)
    else:
        fetch_with_codec(args.codec_clean_path, args.codec_name, args.data_cache, args.train_percent, args.test_percent, args.use_genders, args.annotations_path)