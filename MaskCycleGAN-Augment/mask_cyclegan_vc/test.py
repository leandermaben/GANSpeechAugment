import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchaudio

from mask_cyclegan_vc.model import Generator, Discriminator
from args.cycleGAN_test_arg_parser import CycleGANTestArgParser
from dataset.vc_dataset import VCDataset
from dataset.base_functions import make_dataset
from mask_cyclegan_vc.utils import decode_melspectrogram
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver
import ntpath
from PIL import Image

from mask_cyclegan_vc.utils import denorm_and_numpy, getTimeSeries, extract, power_to_db
import soundfile as sf
from dataset.noise_dataset import NoiseDataset
import time

class MaskCycleGANVCTesting(object):
    """Tester for MaskCycleGAN-VC
    """

    def __init__(self, args):
        """
        Args:
            args (Namespace): Program arguments from argparser
        """
        # Store Args
        self.device = args.device
        self.use_res = args.use_res

        args.num_threads = 0   # test code only supports num_threads = 0
        args.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        args.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        args.max_mask_len = 50 #Does not have any impact since an all 1s mask is used
        
        if hasattr(args,'eval') and args.eval:
            self.eval=True
            os.makedirs(args.eval_save_dir, exist_ok=True)
            self.eval_save_path = os.path.join(args.eval_save_dir,args.filename)
        else:
            self.eval=False
            self.converted_audio_dir = os.path.join(args.save_dir, args.name, 'converted_audio')
            os.makedirs(self.converted_audio_dir, exist_ok=True)

        self.model_name = args.model_name

        self.speaker_A_id = args.speaker_A_id
        self.speaker_B_id = args.speaker_B_id


        self.sample_rate = args.sample_rate

        self.dataset = NoiseDataset(args)

        self.test_dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           drop_last=False)

        # Generator
        in_channels= 3 if self.use_res else 2
        out_channels = 2 if self.use_res else 1
        self.generator = Generator(in_channels=in_channels, out_channels=out_channels).to(self.device)
        self.generator.eval()

        # Load Generator from ckpt
        self.saver = ModelSaver(args)
        self.saver.load_model(self.generator, self.model_name)

        #Getting train stats

        self.A_max = -float('inf')
        self.A_min = float('inf')
        self.B_max = -float('inf')
        self.B_min = float('inf')

        train_A = os.path.join(args.dataroot,args.class_ids[0],'train')
        train_B = os.path.join(args.dataroot,args.class_ids[1],'train')
        A_paths = sorted(make_dataset(train_A, args.max_dataset_size))
        B_paths = sorted(make_dataset(train_B, args.max_dataset_size))

        for path in A_paths:
            mag_spec, phase, sr = extract(path, sr=8000, energy=1.0, state = 'train')
            log_spec = power_to_db(mag_spec)
            self.A_max = max(self.A_max,log_spec.max())
            self.A_min = min(self.A_min,log_spec.min())

        for path in B_paths:
            mag_spec, phase, sr = extract(path, sr=8000, energy=1.0, state = 'train')
            log_spec = power_to_db(mag_spec)
            self.B_max = max(self.B_max,log_spec.max())
            self.B_min = min(self.B_min,log_spec.min())

        
    def save_audio(self, opt, visuals_list, img_path, label):

        """
        Borrowed from https://github.com/shashankshirol/GeneratingNoisySpeechData
        """

        results_dir = os.path.join(opt.save_dir, opt.name)
        img_dir = os.path.join(results_dir, 'audios')
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]

        file_name = '%s/%s.wav' % (label, name)
        os.makedirs(os.path.join(img_dir, label), exist_ok=True)
        save_path = os.path.join(img_dir, file_name)

        flag_first = True

        for visual in visuals_list:
            im_data = visual #Obtaining the generated Output
            im = denorm_and_numpy(im_data.unsqueeze(1)) #De-Normalizing the output tensor to reconstruct the spectrogram

            #Resizing the output to 129x128 size (original splits)
            if(im.shape[-1] == 1): #to drop last channel
                im = im[:,:,0]
            im = Image.fromarray(im)
            im = im.resize((128, 129), Image.LANCZOS)
            im = np.asarray(im).astype(np.float)

            if(flag_first):
                spec = im
                flag_first = False
            else:
                spec = np.concatenate((spec, im), axis=1) #concatenating specs to obtain original.

        if label[-1] == 'A':
            train_max = self.A_max
            train_min = self.A_min
        elif label[-1] == 'B':
            train_max = self.B_max
            train_min = self.B_min
            
        data, sr = getTimeSeries(spec, img_path, opt.spec_power, opt.energy, state = opt.phase, train_min = train_min,train_max =train_max)
        sf.write(save_path, data, sr)

        return

    def loadPickleFile(self, fileName):
        """Loads a Pickle file.

        Args:
            fileName (str): pickle file path

        Returns:
            file object: The loaded pickle file object
        """
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def test(self):

        ds_len = self.dataset.get_clean_len() if self.model_name == 'generator_A2B' else self.dataset.get_noisy_len()
        idx = 0
        datas = []

        for i, data in enumerate(self.test_dataloader):
            datas.append(data)
        while idx < ds_len:

            # if(idx >= args.num_test):
            #     break

            if self.model_name == 'generator_A2B':
                real = datas[idx]['A'].to(self.device, dtype=torch.float)
                mask = datas[idx]['A_mask'].to(self.device, dtype=torch.float)
                img_path = datas[idx]['A_paths']
            else:
                real = datas[idx]['B'].to(self.device, dtype=torch.float)
                mask = datas[idx]['B_mask'].to(self.device, dtype=torch.float)
                img_path = datas[idx]['B_paths']
            if not self.use_res:
                fake = self.generator(real, mask)
            else:
                fake, _ = self.generator(real, mask, torch.zeros_like(real))
            visuals_list = [fake.detach().cpu()]
            num_comps = datas[idx]["A_comps"] if self.model_name == 'generator_A2B' else datas[idx]["B_comps"]
            comps_processed = 1
            

            while(comps_processed < num_comps):
                idx += 1
                if self.model_name == 'generator_A2B':
                    real = datas[idx]['A'].to(self.device, dtype=torch.float)
                    mask = datas[idx]['A_mask'].to(self.device, dtype=torch.float)
                    img_path = datas[idx]['A_paths']
                else:
                    real = datas[idx]['B'].to(self.device, dtype=torch.float)
                    mask = datas[idx]['B_mask'].to(self.device, dtype=torch.float)
                    img_path = datas[idx]['B_paths']
                if not self.use_res:
                    fake = self.generator(real, mask)
                else:
                    fake, _ = self.generator(real, mask, torch.zeros_like(real))
                del real
                del mask
                visuals_list.append(fake.detach().cpu())
                comps_processed += 1

            print("saving: ", img_path[0])
            self.save_audio(args, visuals_list, img_path, label= 'fake_B' if self.model_name == 'generator_A2B' else 'fake_A' )
            idx += 1



if __name__ == "__main__":
    start =time.time()
    parser = CycleGANTestArgParser()
    args = parser.parse_args()
    tester = MaskCycleGANVCTesting(args)
    tester.test()
    end = time.time()
    duration = end-start
    print(f'Time taken is {duration//60} minutes {duration%60} seconds.')

