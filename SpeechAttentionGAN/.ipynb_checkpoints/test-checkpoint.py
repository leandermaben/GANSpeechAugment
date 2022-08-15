"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import ntpath
from PIL import Image
from util.util import denorm_and_numpy, getTimeSeries
import soundfile as sf
import numpy as np
import json

with open('defaults.json','r') as f:
    defaults = json.load(f)

def save_audio(opt, visuals_list, img_path, use_phase=False, label='fakeA'):

    """
    Borrowed from https://github.com/shashankshirol/GeneratingNoisySpeechData

    Modified by: Leander Maben.
    """

    results_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    img_dir = os.path.join(results_dir, 'audios')
    short_path = ntpath.basename(img_path[0])
    name = os.path.splitext(short_path)[0]

    

    file_name = '%s/%s.wav' % (label, name)
    os.makedirs(os.path.join(img_dir, label), exist_ok=True)
    save_path = os.path.join(img_dir, file_name)

    flag_first = True

    for visual in visuals_list:
        im_data = visual[label] #Obtaining the generated Output
        im = denorm_and_numpy(im_data) #De-Normalizing the output tensor to reconstruct the spectrogram

        #Resizing the output to 129x128 size (original splits)
        if(im.shape[-1] == 1): #to drop last channel
            im = im[:,:,0]
        if use_phase:
            im_mag = Image.fromarray(im[:,:,0])
            im_phase = Image.fromarray(im[:,:,1])
            im_mag = im_mag.resize((defaults["fix_w"], defaults["n_fft"]//2+1), Image.LANCZOS)
            im_phase = im_phase.resize((defaults["fix_w"], defaults["n_fft"]//2+1), Image.LANCZOS)
            im_mag = np.asarray(im_mag).astype(np.float)
            im_phase = np.asarray(im_phase).astype(np.float)
            if(flag_first):
                mag_spec = im_mag
                phase_spec = im_phase
                flag_first = False
            else:
                mag_spec = np.concatenate((mag_spec, im_mag), axis=1) #concatenating specs to obtain original.
                phase_spec = np.concatenate((phase_spec, im_phase), axis=1) #concatenating specs to obtain original.
        else:
            im = Image.fromarray(im)
            im = im.resize((defaults["fix_w"], defaults["n_fft"]//2+1), Image.LANCZOS)
            im = np.asarray(im).astype(np.float)

            if(flag_first):
                mag_spec = im
                phase_spec=None
                flag_first = False
            else:
                mag_spec = np.concatenate((mag_spec, im), axis=1) #concatenating specs to obtain original.


    data, sr = getTimeSeries(mag_spec, phase_spec, img_path, opt.spec_power, opt.energy, state = opt.phase, use_phase=use_phase)
    print('Output')
    print(data.shape)
    sf.write(save_path, data, sr)

    return


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    """
    Borrowed from https://github.com/shashankshirol/GeneratingNoisySpeechData
    """

    ## A -> B

    ds_len = dataset.get_A_len()
    idx = 0
    datas = []
    for i, data in enumerate(dataset):
        datas.append(data)
    while idx < ds_len:

        model.set_input(datas[idx])
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        visuals_list = [visuals]
        num_comps = datas[idx]["A_comps"]
        comps_processed = 1

        while(comps_processed < num_comps):
            idx += 1
            model.set_input(datas[idx])
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            visuals_list.append(visuals)
            comps_processed += 1

        print("saving: ", img_path[0])
        save_audio(opt, visuals_list, img_path, use_phase=opt.use_phase, label='fake_B')
        #save_audio(opt, visuals_list, img_path, use_phase=opt.use_phase, label='rec_A')
        idx += 1
    
    # ## B -> A

    # ds_len = dataset.get_B_len()
    # idx = 0
    # datas = []
    # for i, data in enumerate(dataset):
    #     datas.append(data)
    # while idx < ds_len:

    #     model.set_input(datas[idx])
    #     model.test()
    #     visuals = model.get_current_visuals()
    #     img_path = datas[idx]['B_paths']
    #     print(img_path)
    #     visuals_list = [visuals]
    #     num_comps = datas[idx]["B_comps"]
    #     print(num_comps)
    #     comps_processed = 1

    #     while(comps_processed < num_comps):
    #         idx += 1
    #         model.set_input(datas[idx])
    #         model.test()
    #         visuals = model.get_current_visuals()
    #         img_path = datas[idx]["B_paths"]
    #         visuals_list.append(visuals)
    #         comps_processed += 1

    #     print("saving: ", img_path[0])
    #     save_audio(opt, visuals_list, img_path, use_phase=opt.use_phase, label='fake_A')
    #     idx += 1
        
