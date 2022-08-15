"""
Trains MaskCycleGAN-VC as described in https://arxiv.org/pdf/2102.12841.pdf
Inspired by https://github.com/jackaduma/CycleGAN-VC2
"""

import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as data

from mask_cyclegan_vc.model import Generator, Discriminator
from args.cycleGAN_train_arg_parser import CycleGANTrainArgParser
from dataset.noise_dataset import NoiseDataset
from mask_cyclegan_vc.utils import decode_melspectrogram, get_mel_spectrogram_fig
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver
from mask_cyclegan_vc.DiffAugment_pythorch import DiffAugment
from mask_cyclegan_vc.patchNCE import patchNCE

class MaskCycleGANVCTraining(object):
    """Trainer for MaskCycleGAN-VC
    """

    def __init__(self, args):
        """
        Args:
            args (Namespace): Program arguments from argparser
        """
        # Store args
        self.num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.generator_lr = args.generator_lr
        self.discriminator_lr = args.discriminator_lr
        self.decay_after = args.decay_after
        self.stop_identity_after = args.stop_identity_after
        self.mini_batch_size = args.batch_size
        self.cycle_loss_lambda = args.cycle_loss_lambda
        self.identity_loss_lambda = args.identity_loss_lambda
        self.patch_loss_lambda = 10 #Add to arg parser
        self.device = args.device
        self.epochs_per_save = args.epochs_per_save
        self.epochs_per_plot = args.epochs_per_plot
        self.policy = 'color,translation,cutout' #Policy for diffAugment
        self.patchNCE = False
        self.diff_aug = False
        self.use_res = args.use_res
        # Initialize MelGAN-Vocoder used to decode Mel-spectrograms
        # self.vocoder = torch.hub.load(
        #     'descriptinc/melgan-neurips', 'load_melgan')
        # self.sample_rate = args.sample_rate

        # # Initialize speakerA's dataset
        # Srtifacts from melGAN
        # self.dataset_A = self.loadPickleFile(os.path.join(
        #     args.preprocessed_data_dir, args.speaker_A_id, f"{args.speaker_A_id}_normalized.pickle"))
        # dataset_A_norm_stats = np.load(os.path.join(
        #     args.preprocessed_data_dir, args.speaker_A_id, f"{args.speaker_A_id}_norm_stat.npz"))
        # self.dataset_A_mean = dataset_A_norm_stats['mean']
        # self.dataset_A_std = dataset_A_norm_stats['std']

        # Initialize speakerB's dataset
        # self.dataset_B = self.loadPickleFile(os.path.join(
        #     args.preprocessed_data_dir, args.speaker_B_id, f"{args.speaker_B_id}_normalized.pickle"))
        # dataset_B_norm_stats = np.load(os.path.join(
        #     args.preprocessed_data_dir, args.speaker_B_id, f"{args.speaker_B_id}_norm_stat.npz"))
        # self.dataset_B_mean = dataset_B_norm_stats['mean']
        # self.dataset_B_std = dataset_B_norm_stats['std']

       
        # Initialize Train Dataloader
        self.num_frames = args.num_frames
        self.dataset = NoiseDataset(args)
        self.train_dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                            batch_size=self.mini_batch_size,
                                                            shuffle=True,
                                                            drop_last=False,
                                                            num_workers=args.num_threads)

         # Compute lr decay rate
        self.n_samples = len(self.dataset)
        print(f'n_samples = {self.n_samples}')
        self.generator_lr_decay = self.generator_lr / \
            float(self.num_epochs * (self.n_samples // self.mini_batch_size))
        self.discriminator_lr_decay = self.discriminator_lr / \
            float(self.num_epochs * (self.n_samples // self.mini_batch_size))
        print(f'generator_lr_decay = {self.generator_lr_decay}')
        print(f'discriminator_lr_decay = {self.discriminator_lr_decay}')


        # Initialize Validation Dataloader (used to generate intermediate outputs)
        self.validation_dataset = NoiseDataset(args,valid=True)
        self.validation_dataloader = torch.utils.data.DataLoader(dataset=self.validation_dataset,
                                                                 batch_size=1,
                                                                 shuffle=False,
                                                                 drop_last=False)

        # Initialize logger and saver objects
        self.logger = TrainLogger(args, len(self.train_dataloader.dataset))
        self.saver = ModelSaver(args)

        # Initialize Generators and Discriminators
        in_channels_gen = 3 if args.use_res else 2
        out_channels_gen = 2 if args.use_res else 1

        self.generator_A2B = Generator(input_shape=((args.crop_size, args.crop_size)),in_channels=in_channels_gen, out_channels=out_channels_gen).to(self.device)
        self.generator_B2A = Generator(input_shape=((args.crop_size, args.crop_size)),in_channels=in_channels_gen, out_channels=out_channels_gen).to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)
        # Discriminator to compute 2 step adversarial loss
        self.discriminator_A2 = Discriminator().to(self.device)
        # Discriminator to compute 2 step adversarial loss
        self.discriminator_B2 = Discriminator().to(self.device)

        # Initialize Optimizers
        g_params = list(self.generator_A2B.parameters()) + \
            list(self.generator_B2A.parameters())
        d_params = list(self.discriminator_A.parameters()) + \
            list(self.discriminator_B.parameters()) + \
            list(self.discriminator_A2.parameters()) + \
            list(self.discriminator_B2.parameters())
        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(
            d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))

        # Load from previous ckpt
        if args.continue_train:
            self.saver.load_model(
                self.generator_A2B, "generator_A2B", None, self.generator_optimizer)
            self.saver.load_model(self.generator_B2A,
                                  "generator_B2A", None, None)
            self.saver.load_model(self.discriminator_A,
                                  "discriminator_A", None, self.discriminator_optimizer)
            self.saver.load_model(self.discriminator_B,
                                  "discriminator_B", None, None)
            self.saver.load_model(self.discriminator_A2,
                                  "discriminator_A2", None, None)
            self.saver.load_model(self.discriminator_B2,
                                  "discriminator_B2", None, None)

    def adjust_lr_rate(self, optimizer, generator):
        """Decays learning rate.

        Args:
            optimizer (torch.optim): torch optimizer
            generator (bool): Whether to adjust generator lr.
        """
        if generator:
            self.generator_lr = max(
                0., self.generator_lr - self.generator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.generator_lr
        else:
            self.discriminator_lr = max(
                0., self.discriminator_lr - self.discriminator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.discriminator_lr

    def reset_grad(self):
        """Sets gradients of the generators and discriminators to zero before backpropagation.
        """
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def loadPickleFile(self, fileName):
        """Loads a Pickle file.

        Args:
            fileName (str): pickle file path

        Returns:
            file object: The loaded pickle file object
        """
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def train(self):
        """Implements the training loop for MaskCycleGAN-VC
        """
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.logger.start_epoch()

            for i, data_point in enumerate(tqdm(self.train_dataloader)):
                
                real_A = data_point['A']
                mask_A = data_point['A_mask']
                real_B = data_point['B']
                mask_B = data_point['B_mask']

                self.logger.start_iter()
                num_iterations = (
                    self.n_samples // self.mini_batch_size) * epoch + i

                with torch.set_grad_enabled(True):
                    real_A = real_A.to(self.device, dtype=torch.float)
                    mask_A = mask_A.to(self.device, dtype=torch.float)
                    real_B = real_B.to(self.device, dtype=torch.float)
                    mask_B = mask_B.to(self.device, dtype=torch.float)
                    if self.use_res:
                        res_A = torch.zeros_like(real_A)
                        res_B = torch.zeros_like(real_B)
                    else:
                        res_A=None
                        res_B=None

                    # ----------------
                    # Train Generator
                    # ----------------
                    self.generator_A2B.train()
                    self.generator_B2A.train()
                    self.discriminator_A.eval()
                    self.discriminator_B.eval()
                    self.discriminator_A2.eval()
                    self.discriminator_B2.eval()

                    # Generator Feed Forward
                    if not self.use_res:
                        fake_B = self.generator_A2B(real_A, mask_A)
                        cycle_A = self.generator_B2A(fake_B, torch.ones_like(fake_B))
                        fake_A = self.generator_B2A(real_B, mask_B)
                        cycle_B = self.generator_A2B(fake_A, torch.ones_like(fake_A))
                        identity_A = self.generator_B2A(
                            real_A, torch.ones_like(real_A))
                        identity_B = self.generator_A2B(
                            real_B, torch.ones_like(real_B))
                    else:
                        fake_B, res_B = self.generator_A2B(real_A, mask_A, torch.zeros_like(real_A))
                        cycle_A,_ = self.generator_B2A(fake_B, torch.ones_like(fake_B), res_B)
                        fake_A, res_A = self.generator_B2A(real_B, mask_B, torch.zeros_like(real_B))
                        cycle_B,_ = self.generator_A2B(fake_A, torch.ones_like(fake_A), res_A)
                        identity_A,_ = self.generator_B2A(
                            real_A, torch.ones_like(real_A), torch.zeros_like(real_A))
                        identity_B,_ = self.generator_A2B(
                            real_B, torch.ones_like(real_B), torch.zeros_like(real_B))
                    # print(f"{'*'*25} Shape of img {fake_A.cpu().detach().numpy().shape}")
                    # print(f"{'*'*25} Shape of augmented image {DiffAugment(fake_A,policy=self.policy).cpu().detach().numpy().shape}")
                    if self.diff_aug:
                        d_fake_A = self.discriminator_A(DiffAugment(fake_A,policy=self.policy))
                        d_fake_B = self.discriminator_B(DiffAugment(fake_B,policy=self.policy))
                    else:
                        d_fake_A = self.discriminator_A(fake_A)
                        d_fake_B = self.discriminator_B(fake_B)

                    # For Two Step Adverserial Loss
                    if self.diff_aug:
                        d_fake_cycle_A = self.discriminator_A2(DiffAugment(cycle_A,policy=self.policy))
                        d_fake_cycle_B = self.discriminator_B2(DiffAugment(cycle_B,policy=self.policy))
                    else:
                        d_fake_cycle_A = self.discriminator_A2(cycle_A)
                        d_fake_cycle_B = self.discriminator_B2(cycle_B)

                    # Generator Cycle Loss
                    cycleLoss = torch.mean(
                        torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))

                    # Generator Identity Loss
                    identityLoss = torch.mean(
                        torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

                    # Generator Loss
                    g_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
                    g_loss_B2A = torch.mean((1 - d_fake_A) ** 2)

                    # Generator Two Step Adverserial Loss
                    generator_loss_A2B_2nd = torch.mean((1 - d_fake_cycle_B) ** 2)
                    generator_loss_B2A_2nd = torch.mean((1 - d_fake_cycle_A) ** 2)

                    #PatchNCE Loss
                    if self.patchNCE:
                        patch_loss_A2B = patchNCE(real_A,fakeB,self.generator_A2B)
                        patch_loss_B2A = patchNCE(real_B,fake_A,self.generator_B2A)

                    # Total Generator Loss
                    if self.patchNCE:
                        g_loss = g_loss_A2B + g_loss_B2A + \
                            generator_loss_A2B_2nd + generator_loss_B2A_2nd + \
                            self.cycle_loss_lambda * cycleLoss + self.identity_loss_lambda * identityLoss +\
                            self.patch_loss_lambda * (patch_loss_A2B+patch_loss_B2A)/2
                    else:
                        g_loss = g_loss_A2B + g_loss_B2A + \
                            generator_loss_A2B_2nd + generator_loss_B2A_2nd + \
                            self.cycle_loss_lambda * cycleLoss + self.identity_loss_lambda * identityLoss 

                    # Backprop for Generator
                    self.reset_grad()
                    g_loss.backward()
                    self.generator_optimizer.step()

                    # ----------------------
                    # Train Discriminator
                    # ----------------------
                    self.generator_A2B.eval()
                    self.generator_B2A.eval()
                    self.discriminator_A.train()
                    self.discriminator_B.train()
                    self.discriminator_A2.train()
                    self.discriminator_B2.train()

                    # Discriminator Feed Forward
                    if not self.use_res:
                        generated_A = self.generator_B2A(real_B, mask_B)
                        cycled_B = self.generator_A2B(
                        generated_A, torch.ones_like(generated_A))
                        generated_B = self.generator_A2B(real_A, mask_A)
                        cycled_A = self.generator_B2A(
                        generated_B, torch.ones_like(generated_B))
                    else:
                        generated_A, generated_res_A = self.generator_B2A(real_B, mask_B, torch.zeros_like(real_B))
                        cycled_B, _ = self.generator_A2B(
                        generated_A, torch.ones_like(generated_A), generated_res_A)
                        generated_B, generated_res_B = self.generator_A2B(real_A, mask_A, torch.zeros_like(real_A))
                        cycled_A,_ = self.generator_B2A(
                        generated_B, torch.ones_like(generated_B), generated_res_B)

                    if self.diff_aug:
                        d_real_A = self.discriminator_A(DiffAugment(real_A ,policy=self.policy))
                        d_real_B = self.discriminator_B(DiffAugment(real_B ,policy=self.policy))
                        d_real_A2 = self.discriminator_A2(DiffAugment(real_A ,policy=self.policy))
                        d_real_B2 = self.discriminator_B2(DiffAugment(real_B ,policy=self.policy))
                        d_fake_A = self.discriminator_A(DiffAugment(generated_A ,policy=self.policy))
                    else:
                        d_real_A = self.discriminator_A(real_A )
                        d_real_B = self.discriminator_B(real_B)
                        d_real_A2 = self.discriminator_A2(real_A)
                        d_real_B2 = self.discriminator_B2(real_B)
                        d_fake_A = self.discriminator_A(generated_A)

                    # For Two Step Adverserial Loss A->B
                    if self.diff_aug:
                        d_cycled_B = self.discriminator_B2(DiffAugment(cycled_B ,policy=self.policy))
                    else:
                        d_cycled_B = self.discriminator_B2(cycled_B)

                    if self.diff_aug:
                        d_fake_B = self.discriminator_B(DiffAugment(generated_B ,policy=self.policy))
                    else:
                        d_fake_B = self.discriminator_B(generated_B)

                    # For Two Step Adverserial Loss B->A
                    if self.diff_aug:
                        d_cycled_A = self.discriminator_A2(DiffAugment(cycled_A ,policy=self.policy))
                    else:
                        d_cycled_A = self.discriminator_A2(cycled_A)

                    # Loss Functions
                    d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
                    d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
                    d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

                    d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
                    d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
                    d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

                    # Two Step Adverserial Loss
                    d_loss_A_cycled = torch.mean((0 - d_cycled_A) ** 2)
                    d_loss_B_cycled = torch.mean((0 - d_cycled_B) ** 2)
                    d_loss_A2_real = torch.mean((1 - d_real_A2) ** 2)
                    d_loss_B2_real = torch.mean((1 - d_real_B2) ** 2)
                    d_loss_A_2nd = (d_loss_A2_real + d_loss_A_cycled) / 2.0
                    d_loss_B_2nd = (d_loss_B2_real + d_loss_B_cycled) / 2.0

                    # Final Loss for discriminator with the Two Step Adverserial Loss
                    d_loss = (d_loss_A + d_loss_B) / 2.0 + \
                        (d_loss_A_2nd + d_loss_B_2nd) / 2.0

                    # Backprop for Discriminator
                    self.reset_grad()
                    d_loss.backward()
                    self.discriminator_optimizer.step()

                # Log Iteration
                self.logger.log_iter(
                    loss_dict={'g_loss': g_loss.item(), 'd_loss': d_loss.item()})
                self.logger.end_iter()

                # Adjust learning rates
                if self.logger.global_step > self.decay_after:
                    self.adjust_lr_rate(
                        self.generator_optimizer, generator=True)
                    self.adjust_lr_rate(
                        self.generator_optimizer, generator=False)

                # Set identity loss to zero if larger than given value
                if self.logger.global_step > self.stop_identity_after:
                    self.identity_loss_lambda = 0

            # Log intermediate outputs on Tensorboard
            # if self.logger.epoch % self.epochs_per_plot == 0:
            #     with torch.no_grad():
            #         # Log Mel-spectrograms .png
            #         real_mel_A_fig = get_mel_spectrogram_fig(
            #             real_A[0].detach().cpu())
            #         fake_mel_A_fig = get_mel_spectrogram_fig(
            #             generated_A[0].detach().cpu())
            #         real_mel_B_fig = get_mel_spectrogram_fig(
            #             real_B[0].detach().cpu())
            #         fake_mel_B_fig = get_mel_spectrogram_fig(
            #             generated_B[0].detach().cpu())
            #         self.logger.visualize_outputs({"real_A_spec": real_mel_A_fig, "fake_B_spec": fake_mel_B_fig,
            #                                        "real_B_spec": real_mel_B_fig, "fake_A_spec": fake_mel_A_fig})

            #         # Convert Mel-spectrograms from validation set to waveform and log to tensorboard
            #         real_mel_full_A, real_mel_full_B = next(
            #             iter(self.validation_dataloader))
            #         real_mel_full_A = real_mel_full_A.to(
            #             self.device, dtype=torch.float)
            #         real_mel_full_B = real_mel_full_B.to(
            #             self.device, dtype=torch.float)
            #         fake_mel_full_B = self.generator_A2B(
            #             real_mel_full_A, torch.ones_like(real_mel_full_A))
            #         fake_mel_full_A = self.generator_B2A(
            #             real_mel_full_B, torch.ones_like(real_mel_full_B))
            #         real_wav_full_A = decode_melspectrogram(self.vocoder, real_mel_full_A[0].detach(
            #         ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
            #         fake_wav_full_A = decode_melspectrogram(self.vocoder, fake_mel_full_A[0].detach(
            #         ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
            #         real_wav_full_B = decode_melspectrogram(self.vocoder, real_mel_full_B[0].detach(
            #         ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()
            #         fake_wav_full_B = decode_melspectrogram(self.vocoder, fake_mel_full_B[0].detach(
            #         ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()
            #         self.logger.log_audio(
            #             real_wav_full_A.T, "real_speaker_A_audio", self.sample_rate)
            #         self.logger.log_audio(
            #             fake_wav_full_A.T, "fake_speaker_A_audio", self.sample_rate)
            #         self.logger.log_audio(
            #             real_wav_full_B.T, "real_speaker_B_audio", self.sample_rate)
            #         self.logger.log_audio(
            #             fake_wav_full_B.T, "fake_speaker_B_audio", self.sample_rate)

            # Save each model checkpoint
            if self.logger.epoch % self.epochs_per_save == 0:
                self.saver.save(self.logger.epoch, self.generator_A2B,
                                self.generator_optimizer, None, args.device, "generator_A2B")
                self.saver.save(self.logger.epoch, self.generator_B2A,
                                self.generator_optimizer, None, args.device, "generator_B2A")
                self.saver.save(self.logger.epoch, self.discriminator_A,
                                self.discriminator_optimizer, None, args.device, "discriminator_A")
                self.saver.save(self.logger.epoch, self.discriminator_B,
                                self.discriminator_optimizer, None, args.device, "discriminator_B")
                self.saver.save(self.logger.epoch, self.discriminator_A2,
                                self.discriminator_optimizer, None, args.device, "discriminator_A2")
                self.saver.save(self.logger.epoch, self.discriminator_B2,
                                self.discriminator_optimizer, None, args.device, "discriminator_B2")

            self.logger.end_epoch()


if __name__ == "__main__":
    parser = CycleGANTrainArgParser()
    args = parser.parse_args()
    cycleGAN = MaskCycleGANVCTraining(args)
    cycleGAN.train()
