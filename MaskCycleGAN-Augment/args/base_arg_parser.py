"""
Base arguments for all scripts.
"""

import argparse
import json
import os
import torch
import numpy as np
import random

DATAROOT_DEFAULT = '/content/MaskCycleGAN-Augment/data_cache'

class BaseArgParser(object):
    """
    Base argument parser for args shared between test and train modes.

    ...
    Attributes
    ----------
    parser : argparse.ArgumentParser
        ArgumentParser object used to parse command line args

    Methods
    -------
    parse_args():
        Parse arguments, create checkpoints and vizualization directory, and sets up gpu device

    print_options(args):
        Prints and save options
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(description=' ')
        self.isTrain = False

        self.parser.add_argument(
            '--name', type=str, default='debug', help='Experiment name prefix.')
        self.parser.add_argument(
            '--batch_size', type=int, default=20, help='Batch size.')
        self.parser.add_argument(
            '--save_dir', type=str, default='/home/results/', help='Directory for results including ckpts.')
        self.parser.add_argument(
            '--seed', type=int, default=0, help='Random Seed.')
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='Comma-separated list of GPU IDs.')
        self.parser.add_argument('--use_res', action='store_true',
                                 help='Generate residual information')

        # Logger Args
        self.parser.add_argument('--steps_per_print', type=int, default=1000,
                                 help='Number of steps between printing loss to the console and TensorBoard.')
        self.parser.add_argument('--epochs_per_save', type=int, default=1,
                                 help='Number of epochs between saving the model.')
        self.parser.add_argument(
            '--start_epoch', type=int, default=1, help='Epoch to start training')
        self.parser.add_argument('--load_epoch', type=int, default=0,
                                 help='Default uses latest cached model if continue train or eval set')
        
        #Dataset args
        self.parser.add_argument('--class_ids', dest='class_ids', type=str, default=['clean','noisy'], help='class IDS of the two domains.')
        self.parser.add_argument('--spec_power', dest='spec_power', type=float, default=1.0, help='Number to raise spectrogram by.')
        self.parser.add_argument('--energy', dest='energy', type=float, default=1.0, help='to modify the energy/amplitude of the audio-signals')
        self.parser.add_argument('--dataroot', dest='dataroot', type=str, default=DATAROOT_DEFAULT, help="Directory with data.")
        self.parser.add_argument('--split', dest='split', type=str, default='use_default', help="Split to use for data set. If data has not been split into sets pass None.")
        self.parser.add_argument('--data_load_order', dest='data_load_order', default='use_default', type=str,choices=['aligned','unaligned'], help="Load Data as aligned or unaligned. For test phase it is unaligned aligned by default and for train it is unaligned by default.")
        self.parser.add_argument('--load_size', type=int, default=128, help='scale images to this size')
        self.parser.add_argument('--preprocess', type=str, default='resize', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none | passcodec]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--crop_size', dest='crop_size', type=int, default=128, help='Size after cropping')
        self.parser.add_argument('--max_dataset_size', dest='max_dataset_size', type=int, default=float('inf'), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--num_threads', dest='num_threads', type=int, default=4, help='Number of threads for dataloader')

    def parse_args(self):
        """
        Function that parses arguments, create checkpoints and vizualization directory, and sets up gpu device.

        Returns
        -------
        args : Namespace
            Parsed program arguments
        """
        args = self.parser.parse_args()

        # Limit sources of nondeterministic behavior
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

        if hasattr(self, 'isTrain'):
            args.isTrain = self.isTrain   # train or test

        os.makedirs(os.path.join(args.save_dir, args.name), exist_ok=True)

        # Save args to a JSON file
        prefix = 'train' if args.isTrain else 'test'
        with open(os.path.join(args.save_dir, args.name, f"{prefix}_args.json"), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')

        # Create ckpt dir and viz dir
        if args.isTrain:
            args.ckpt_dir = os.path.join(args.save_dir, args.name, 'ckpts')
            os.makedirs(args.ckpt_dir, exist_ok=True)

        # Set up available GPUs
        def args_to_list(csv, arg_type=int):
            """Convert comma-separated arguments to a list."""
            arg_vals = [arg_type(d) for d in str(csv).split(',')]
            return arg_vals

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        args.gpu_ids = args_to_list(args.gpu_ids)

        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(0)
            args.gpu_ids = ['cuda' + ':' + str(i)
                            for i in range(len(args.gpu_ids))]
            args.device = 'cuda'
        else:
            args.device = 'cpu'

        # Ensure consistency of load_epoch and start_epoch arguments with each other and defaults.
        if not args.isTrain or (hasattr(args, 'continue_train') and args.continue_train):
            if args.load_epoch > 0:
                args.start_epoch = args.load_epoch + 1
            elif args.start_epoch > 1:
                args.load_epoch = args.start_epoch - 1
            else:
                args.load_epoch = self.get_last_saved_epoch(args)
                args.start_epoch = args.load_epoch + 1

        args.phase = 'train' if args.isTrain else 'test' 
        args.no_flip = True if args.isTrain else args.no_flip
        if args.split == 'None':
            args.split=''
        elif args.split == 'use_default':
            args.split = args.phase
        
        if args.data_load_order =='use_default':
            args.data_load_order = 'unaligned' if args.isTrain else 'aligned'




        self.print_options(args)



        return args

    def get_last_saved_epoch(self, args):
        """Returns the last epoch at which a checkpoint was saved.

        Parameters
        ----------
        args : Namespace
            Arguments for models and model testing

        Returns
        -------
        epoch : int
            Last epoch at which checkpoints were saved
        """
        ckpt_files = sorted([name for name in os.listdir(
            args.ckpt_dir) if name.split(".", 1)[1] == "pth.tar"])

        if len(ckpt_files) > 0:
            epoch = int(ckpt_files[-1][:5])
        else:
            epoch = 0
        return epoch

    def print_options(self, args):
        """
        Function that prints and save options
        It will print both current options and default values(if different).
        Inspired by:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py#L88-L111

        Parameters
        ----------
        args : Namespace
            Arguments for models and model testing
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)
