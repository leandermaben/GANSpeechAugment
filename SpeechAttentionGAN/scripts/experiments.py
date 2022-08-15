import os
import pandas as pd
import shutil
import sys
from metrics.evaluate_lsd import main as lsd
from metrics.mssl import main as mssl

def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)

# def log(path,name,comment,avg_lsd,std_lsd,avg_mssl,std_mssl,male_avg_lsd,male_std_lsd,male_avg_mssl,male_std_mssl,female_avg_lsd,female_std_lsd,female_avg_mssl,female_std_mssl):
#     """
#     Created by Leander Maben.
#     """
#     df=pd.read_csv(path)
#     df.loc[len(df.index)] = {'name':name,'comment':comment,'avg_lsd':avg_lsd,'std_lsd':std_lsd,'avg_mssl':avg_mssl,'std_mssl':std_mssl,'male_avg_lsd':male_avg_lsd,'male_std_lsd':male_std_lsd,'male_avg_mssl':male_avg_mssl,'male_std_mssl':male_std_mssl,'female_avg_lsd':female_avg_lsd,'female_std_lsd':female_std_lsd,'female_avg_mssl':female_avg_mssl,'female_std_mssl':female_std_mssl}
#     df.to_csv(path,index=False)

# def log(path,name,comment,avg_lsd,std_lsd,avg_mssl,std_mssl):
#     """
#     Created by Leander Maben.
#     """
#     df=pd.read_csv(path)
#     df.loc[len(df.index)] = {'name':name,'comment':comment,'avg_lsd':avg_lsd,'std_lsd':std_lsd,'avg_mssl':avg_mssl,'std_mssl':std_mssl}
#     df.to_csv(path,index=False)

def log(path,info):
    """
    Created by Leander Maben.
    """
    df=pd.read_csv(path)
    df.loc[len(df.index)] = info
    df.to_csv(path,index=False)

def validate(name, epochs, data_cache, results_dir):
    info = {'avg_val_lsd':[],'std_val_lsd':[],'avg_val_mssl':[],'std_val_mssl':[]}
    min_lsd=sys.maxsize
    min_mssl=sys.maxsize
    min_lsd_epoch=-1
    min_mssl_epoch=-1

    for epoch in epochs:
        run(f'python test.py --dataroot data_cache --phase val --name {name} --model attention_gan --dataset_mode audio --norm instance --phase test --no_dropout --load_size_h 128 --load_size_w 128 --crop_size 128 --batch_size 1 --gpu_ids 0 --input_nc 1 --output_nc 1 --use_mask --checkpoints_dir /content/drive/MyDrive/APSIPA/Results/checkpoints --epoch {epoch} --results_dir {results_dir}')
        avg_lsd,std_lsd = lsd(os.path.join(data_cache,'noisy','val'),os.path.join(results_dir,name,f'val_{epoch}','audios','fake_B'),use_gender=False)
        avg_mssl,std_mssl = mssl(os.path.join(data_cache,'noisy','val'),os.path.join(results_dir,name,f'val_{epoch}','audios','fake_B'),use_gender=False)

        info['avg_val_lsd'].append(avg_lsd)
        info['std_val_lsd'].append(std_lsd)
        info['avg_val_mssl'].append(avg_mssl)
        info['std_val_mssl'].append(std_mssl)

        if avg_lsd<min_lsd:
            min_lsd=avg_lsd
            min_lsd_epoch=epoch
        
        if avg_mssl<min_mssl:
            min_mssl=avg_mssl
            min_mssl_epoch=epoch
        
    info['min_val_lsd'] = min_lsd
    info['min_val_mssl'] = min_mssl
    info['min_lsd_epoch'] = min_lsd_epoch
    info['min_mssl_epoch'] = min_mssl_epoch

    return info


def apsipa_exp(names,csv_path,sources, data_cache='/content/AttentionGAN-VC/data_cache',results_dir='/content/AttentionGAN-VC/results', epochs = [50,100,150,200,250,300,350,400]):
     for name, source in zip(names,sources):
        print('#'*25)
        print(f'Training {name} with Data from {source}')
        shutil.copytree(os.path.join('/content/drive/MyDrive/APSIPA/Data_Sources',source),data_cache)
        #run(f'python train.py --dataroot data_cache --name {name} --model attention_gan --dataset_mode audio --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size_h 128 --load_size_w 128 --crop_size 128 --preprocess resize --batch_size 4 --niter 200 --niter_decay 200 --gpu_ids 0 --display_id 0 --display_freq 100 --print_freq 100 --input_nc 1 --output_nc 1 --use_cycled_discriminators --use_mask --max_mask_len 50 --checkpoints_dir /content/drive/MyDrive/APSIPA/Results/checkpoints --no_html')
        
        info = validate(name, epochs, data_cache, results_dir)
        for metric in ['min_lsd_epoch','min_mssl_epoch']:
            epoch = info[metric]
            run(f'python test.py --dataroot data_cache --name {name} --model attention_gan --dataset_mode audio --norm instance --phase test --no_dropout --load_size_h 128 --load_size_w 128 --crop_size 128 --batch_size 1 --gpu_ids 0 --input_nc 1 --output_nc 1 --use_mask --checkpoints_dir /content/drive/MyDrive/APSIPA/Results/checkpoints --epoch {epoch} --results_dir {results_dir}')
            info[f'avg_test_lsd_{metric}'],info[f'std_test_lsd_{metric}'] = lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,f'test_{epoch}','audios','fake_B'),use_gender=False)
            info[f'avg_test_mssl_{metric}'],info[f'std_test_mssl_{metric}'] = mssl(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,f'test_{epoch}','audios','fake_B'),use_gender=False)

            if info['min_mssl_epoch'] == info['min_lsd_epoch']:
                info['avg_test_lsd_min_mssl_epoch'] = info['avg_test_lsd_min_lsd_epoch']
                info['std_test_lsd_min_mssl_epoch'] = info['std_test_lsd_min_lsd_epoch']
                info['avg_test_mssl_min_mssl_epoch'] = info['avg_test_mssl_min_lsd_epoch']
                info['std_test_mssl_min_mssl_epoch'] = info['std_test_mssl_min_lsd_epoch']
                break

        info['name'] = name
        info['comment'] = f'AttentionGAN-VC trained for 200 epochs+ 200 with decay on {source} dataset'
        log(csv_path, info)

        shutil.rmtree(data_cache)
        print(f'Finished experiment with {name}')
        print('#'*25)


def timit_exp1(names,csv_path,noise_dBs, data_cache='/content/AttentionGAN-VC/data_cache',results_dir='/content/AttentionGAN-VC/results'):
     for name, noise_dB in zip(names,noise_dBs):
        print('#'*25)
        print(f'Training {name} with SNR {noise_dB}')

        run(f'python datasets/fetchData.py --transfer_mode timit --noise_dB {noise_dB}')
        run(f'python train.py --dataroot data_cache --name {name} --model attention_gan --dataset_mode audio --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size_h 128 --load_size_w 128 --crop_size 128 --preprocess resize --batch_size 4 --niter 200 --niter_decay 0 --gpu_ids 0 --display_id 0 --display_freq 100 --print_freq 100 --input_nc 1 --output_nc 1 --use_cycled_discriminators --use_mask --max_mask_len 50 --checkpoints_dir /content/drive/MyDrive/TASLP/EXP1/checkpoints --no_html')
        run(f'python test.py --dataroot data_cache --name {name} --model attention_gan --dataset_mode audio --norm instance --phase test --no_dropout --load_size_h 128 --load_size_w 128 --crop_size 128 --batch_size 1 --gpu_ids 0 --input_nc 1 --output_nc 1 --use_mask --checkpoints_dir /content/drive/MyDrive/TASLP/EXP1/checkpoints')
        avg_lsd,std_lsd= lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=False)
        avg_mssl,std_mssl = mssl(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=False)
        log(csv_path, name,f'Training {name} for EXP1 with SNR {noise_dB} for 200 epochs with cycled_disc, WITHOUT phase and mask. LambdaA & B 10 , lambda_identity 0.5',avg_lsd,std_lsd,avg_mssl,std_mssl)
    
        shutil.rmtree(data_cache)
        print(f'Finished experiment with {name}')
        print('#'*25)

if __name__ == '__main__':
    csv_path = '/content/drive/MyDrive/APSIPA/Results/logs.csv'
    if not os.path.exists(csv_path):
        cols=['name','comment','min_lsd_epoch','min_mssl_epoch','avg_test_lsd_min_lsd_epoch','avg_test_mssl_min_lsd_epoch',	'avg_test_lsd_min_mssl_epoch','avg_test_mssl_min_mssl_epoch','min_val_lsd','min_val_mssl','std_test_lsd_min_lsd_epoch','std_test_mssl_min_lsd_epoch','std_test_lsd_min_mssl_epoch',	'std_test_mssl_min_mssl_epoch','avg_val_lsd','avg_val_mssl','std_val_lsd','std_val_mssl']
        df=pd.DataFrame(columns=cols)
        df.to_csv(csv_path,index=False)
    

    sources = ['Non-Parallel/RATS']
    apsipa_exp([f'AttentionGAN_nophase_mask_cycdisc_{i}' for i in ['np_rats']],csv_path,sources)
    


# def apsipa_exp(names,csv_path,sources, data_cache='/content/AttentionGAN-VC/data_cache',results_dir='/content/AttentionGAN-VC/results'):
#      for name, source in zip(names,sources):
#         print('#'*25)
#         print(f'Training {name} with Data from {source}')
#         shutil.copytree(os.path.join('/content/drive/MyDrive/APSIPA/Data_Sources',source),data_cache)
#         run(f'python train.py --dataroot data_cache --name {name} --model attention_gan --dataset_mode audio --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size_h 128 --load_size_w 128 --crop_size 128 --preprocess resize --batch_size 4 --niter 200 --niter_decay 0 --gpu_ids 0 --display_id 0 --display_freq 100 --print_freq 100 --input_nc 1 --output_nc 1 --use_cycled_discriminators --use_mask --max_mask_len 50 --checkpoints_dir /content/drive/MyDrive/APSIPA/Results/checkpoints --no_html')
#         run(f'python test.py --dataroot data_cache --name {name} --model attention_gan --dataset_mode audio --norm instance --phase test --no_dropout --load_size_h 128 --load_size_w 128 --crop_size 128 --batch_size 1 --gpu_ids 0 --input_nc 1 --output_nc 1 --use_mask --checkpoints_dir /content/drive/MyDrive/APSIPA/Results/checkpoints')
#         avg_lsd,std_lsd= lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=False)
#         avg_mssl,std_mssl = mssl(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=False)
#         log(csv_path, name,f'Training {name} with Dataet source {source} for 200 epochs with cycled_disc, WITHOUT phase and WITH mask. LambdaA & B 10 , lambda_identity 0.5',avg_lsd,std_lsd,avg_mssl,std_mssl)
#         shutil.rmtree(data_cache)
#         print(f'Finished experiment with {name}')
#         print('#'*25)

# def apsipa_exp_parallel(names,csv_path,sources, data_cache='/content/AttentionGAN-VC/data_cache',results_dir='/content/AttentionGAN-VC/results'):
#      for name, source in zip(names,sources):
#         print('#'*25)
#         print(f'Training {name} with Data from {source}')
#         shutil.copytree(os.path.join('/content/drive/MyDrive/APSIPA/Data_Sources',source),data_cache)
#         run(f'python train.py --dataroot data_cache --name {name} --data_load_order aligned --model attention_gan --dataset_mode audio --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size_h 128 --load_size_w 128 --crop_size 128 --preprocess resize --batch_size 4 --niter 200 --niter_decay 0 --gpu_ids 0 --display_id 0 --display_freq 100 --print_freq 100 --input_nc 1 --output_nc 1 --use_cycled_discriminators --use_mask --max_mask_len 50 --checkpoints_dir /content/drive/MyDrive/APSIPA/Results/checkpoints --no_html')
#         run(f'python test.py --dataroot data_cache --name {name} --model attention_gan --dataset_mode audio --norm instance --phase test --no_dropout --load_size_h 128 --load_size_w 128 --crop_size 128 --batch_size 1 --gpu_ids 0 --input_nc 1 --output_nc 1 --use_mask --checkpoints_dir /content/drive/MyDrive/APSIPA/Results/checkpoints')
#         avg_lsd,std_lsd= lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=False)
#         avg_mssl,std_mssl = mssl(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=False)
#         log(csv_path, name,f'Training {name} with Dataet source {source} for 200 epochs with cycled_disc, WITHOUT phase and WITH mask. LambdaA & B 10 , lambda_identity 0.5',avg_lsd,std_lsd,avg_mssl,std_mssl)
#         shutil.rmtree(data_cache)
#         print(f'Finished experiment with {name}')
#         print('#'*25)


# ##TODO: Modify functions to handle male,female,std dev

# ## Experiment with data quantity
# def vary_data(train_percents, test_percent, names, csv_path, batch_size=4, use_cycled_discriminators=True, use_phase=False, use_mask=False, lambda_cyc=10, lambda_identity=0.5, n_epochs=100, n_epochs_decay=0, data_cache='/content/AttentionGAN-VC/data_cache',results_dir='/content/AttentionGAN-VC/results'):
#     """
#     Experiment with different size training sets and document lsd and mssl scores.

#     Created by Leander Maben.
#     """

#     input_nc = 1
#     output_nc = 1

#     bool_args_train = ''
#     bool_args_test = ''

#     if use_cycled_discriminators:
#         bool_args_train+= ' --use_cycled_discriminators'

#     if use_phase:
#         bool_args_train+= ' --use_phase'
#         bool_args_test+= ' --use_phase'
#         input_nc = 2
#         output_nc = 2

#     if use_mask:
#         bool_args_train+= ' --use_mask'
#         bool_args_test+= ' --use_mask'

#     for train_p, name in zip(train_percents,names):
#         print('#'*25)
#         print(f'Training {name} with train_percent {train_p}')

#         run(f'python datasets/fetchData.py --train_percent {train_p} --test_percent {test_percent} --use_genders None')
#         run(f'python train.py --dataroot {data_cache} --name {name} --model attention_gan --dataset_mode audio --pool_size 50 --no_dropout --norm instance --lambda_A {lambda_cyc} --lambda_B {lambda_cyc} --lambda_identity {lambda_identity} --load_size {128} --crop_size {128} --batch_size {batch_size} --niter {n_epochs} --niter_decay {n_epochs_decay} --gpu_ids 0 --display_id 0 --display_freq 100 --print_freq 100 --input_nc {input_nc} --output_nc {output_nc}'+bool_args_train)
#         run(f'python test.py --dataroot {data_cache} --name {name} --model attention_gan --dataset_mode audio --norm instance --phase test --no_dropout --load_size {128} --crop_size {128} --batch_size 1 --gpu_ids 0 --input_nc {input_nc} --output_nc {output_nc}'+bool_args_test)
#         avg_lsd,std_lsd,male_avg_lsd,male_std_lsd,female_avg_lsd,female_std_lsd = lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=True)
#         avg_mssl,std_mssl,male_avg_mssl,male_std_mssl,female_avg_mssl,female_std_mssl = mssl(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=True)
#         log(csv_path, name,f'Training with train_percent {train_p} for {n_epochs} without decay and {n_epochs_decay} with decay and {bool_args_train} and cycled_disc {use_cycled_discriminators},lambda_cyc {lambda_cyc} , lambda_identity {lambda_identity} ,new LANCOZ resize and 128 load and crop size {bool_args_train}',avg_lsd,std_lsd,avg_mssl,std_mssl,male_avg_lsd,male_std_lsd,male_avg_mssl,male_std_mssl,female_avg_lsd,female_std_lsd,female_avg_mssl,female_std_mssl)
    
#         shutil.rmtree(data_cache)
#         print(f'Finished experiment with {name}')
#         print('#'*25)
        
# ## Experiment with data quantity
# def use_codec_data(codec_name, name, csv_path, train_percent=10, test_percent=15, batch_size=4, use_cycled_discriminators=True, use_phase=False, use_mask=False, lambda_cyc=1, lambda_identity=0.5, n_epochs=100, n_epochs_decay=0, data_cache='/content/AttentionGAN-VC/data_cache',results_dir='/content/AttentionGAN-VC/results'):
#     """
#     Experiment with different size training sets and document lsd and mssl scores.

#     Created by Leander Maben.
#     """

#     input_nc = 1
#     output_nc = 1

#     bool_args_train = ''
#     bool_args_test = ''

#     if use_cycled_discriminators:
#         bool_args_train+= ' --use_cycled_discriminators'

#     if use_phase:
#         bool_args_train+= ' --use_phase'
#         bool_args_test+= ' --use_phase'
#         input_nc = 2
#         output_nc = 2

#     if use_mask:
#         bool_args_train+= ' --use_mask'
#         bool_args_test+= ' --use_mask'

#     print('#'*25)
#     print(f'Training {name} with codec {codec_name}')

#     run(f'python datasets/fetchData.py --train_percent {train_percent} --test_percent {test_percent} --use_genders None --transfer_mode codec --codec_name {codec_name}')
#     run(f'python train.py --dataroot {data_cache} --name {name} --model attention_gan --dataset_mode audio --pool_size 50 --no_dropout --norm instance --lambda_A {lambda_cyc} --lambda_B {lambda_cyc} --lambda_identity {lambda_identity} --load_size {128} --crop_size {128} --batch_size {batch_size} --niter {n_epochs} --niter_decay {n_epochs_decay} --gpu_ids 0 --display_id 0 --display_freq 100 --print_freq 100 --input_nc {input_nc} --output_nc {output_nc}'+bool_args_train)
#     run(f'python test.py --dataroot {data_cache} --name {name} --model attention_gan --dataset_mode audio --norm instance --phase test --no_dropout --load_size {128} --crop_size {128} --batch_size 1 --gpu_ids 0 --input_nc {input_nc} --output_nc {output_nc}'+bool_args_test)
#     avg_lsd,std_lsd,male_avg_lsd,male_std_lsd,female_avg_lsd,female_std_lsd = lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=True)
#     avg_mssl,std_mssl,male_avg_mssl,male_std_mssl,female_avg_mssl,female_std_mssl = mssl(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'),use_gender=True)
#     log(csv_path, name,f'Training with train_percent {train_percent} for {n_epochs} without decay and {n_epochs_decay} with decay and {bool_args_train} and cycled_disc {use_cycled_discriminators},lambda_cyc {lambda_cyc} , lambda_identity {lambda_identity} ,new LANCOZ resize and 128 load and crop size {bool_args_train}, using codec {codec_name}',avg_lsd,std_lsd,avg_mssl,std_mssl,male_avg_lsd,male_std_lsd,male_avg_mssl,male_std_mssl,female_avg_lsd,female_std_lsd,female_avg_mssl,female_std_mssl)

#     shutil.rmtree(data_cache)
#     print(f'Finished experiment with {name}')
#     print('#'*25)

# ## Experiment with lambda_L1
# def vary_lambdaL1(lamba_L1s,train_percent ,test_percent, names, csv_path, n_epochs=75, n_epochs_decay=25, data_cache='/content/Pix2Pix-VC/data_cache',results_dir='/content/Pix2Pix-VC/results'):
#     """
#     Experiment with different lambda_L1s and document lsd scores.

#     Created by Leander Maben.
#     """
    
#     for lambd, name in zip(lamba_L1s,names):
#         print('#'*25)
#         print(f'Training {name} with Lambda_l1 {lambd}')
#         run(f'python datasets/fetchData.py --train_percent {train_percent} --test_percent {test_percent}')
#         run(f'python train.py --dataroot {data_cache} --name {name} --model pix2pix --n_epochs {n_epochs} --n_epochs_decay {n_epochs_decay} --direction AtoB --input_nc 1 --output_nc 1 --lambda_L1 {lambd} --netG resnet_9blocks   --dataset_mode audio')
#         run(f'python test.py --dataroot {data_cache} --name {name} --model pix2pix --direction AtoB --netG resnet_9blocks  --input_nc 1 --output_nc 1 --dataset_mode audio')
#         avg_lsd, min_lsd = lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'))
#         log(csv_path, name,f'Training with {lambd} lambda_L1 for {n_epochs} without decay and {n_epochs_decay} with decay',avg_lsd,min_lsd)
#         shutil.rmtree(data_cache)
#         print(f'Finished experiment with {name}')
#         print('#'*25)
    
# def run_eval(checkpoints_dir='/content/Pix2Pix-VC/checkpoints', data_cache='/content/Pix2Pix-VC/data_cache', results_dir='/content/Pix2Pix-VC/results'):
#     """
#     Run evaluation on all stored models

#     Created by Leander Maben
#     """
#     #run(f'python datasets/fetchData.py --train_percent {10} --test_percent {15}')
#     for name in os.listdir(checkpoints_dir):       
#         run(f'python test.py --dataroot {data_cache} --name {name} --model pix2pix --direction AtoB --netG resnet_9blocks  --input_nc 1 --output_nc 1 --dataset_mode audio')
#         avg_lsd, min_lsd = lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'test_latest','audios','fake_B'))



# if __name__ == '__main__':
#     csv_path = '/content/drive/MyDrive/APSIPA/Results/logs.csv'
#     if not os.path.exists(csv_path):
#         cols=['name','comment','avg_lsd','std_lsd','avg_mssl','std_mssl']
#         df=pd.DataFrame(columns=cols)
#         df.to_csv(csv_path,index=False)
    

#     sources = ['Parallel/TIMIT_Helicopter','Parallel/TIMIT_Cabin','Parallel/Codec2']
#     apsipa_exp_parallel([f'AttentionGAN_nophase_mask_cycdisc_{i}' for i in ['np_helicopter','np_cabin','np_cd2']],csv_path,sources)

    #sources = ['Non-Parallel/TIMIT_Helicopter','Non-Parallel/TIMIT_Cabin','Non-Parallel/Codec2']
    #apsipa_exp([f'AttentionGAN_nophase_mask_cycdisc_{i}' for i in ['np_helicopter','np_cabin','np_cd2']],csv_path,sources)
    #noise_dBs = [-3,-6]
    
    #timit_exp1([f'AttentionGAN_nophase_mask_cycdisc_{i}dB' for i in noise_dBs],csv_path,noise_dBs)
    
    #log(csv_path,'Dummy', "Logging default parameters used - 200 lambda_L1, 15% test size",0,0)

    # train_percents =[5,10]
    # vary_data(train_percents,15,[f'att_noisy_cyc_disc_lambda_cyc_5_phase_{i}' for i in train_percents], csv_path,lambda_cyc=5, use_phase=True)

    # train_percents =[5,10,25]
    # vary_data(train_percents,15,[f'att_noisy_cyc_disc_lambda_cyc_1_mask_{i}' for i in train_percents], csv_path,lambda_cyc=1, use_mask=True)

    
    # for train_p in [10,25,50]:
    #     use_codec_data('codec2', f'att_codec2_{train_p}',csv_path)

    # train_percents =[10]
    # vary_data(train_percents,15,[f'att_noisy_cyc_disc_lambda_cyc_1_{i}' for i in train_percents], csv_path,lambda_cyc=1, use_cycled_discriminators=False)
    

    # train_percents =[5,10,25]
    # vary_data(train_percents,15,[f'att_noisy_cyc_disc_{i}' for i in train_percents], csv_path)
    #train_percents =[5,10,25]
    # vary_data(train_percents,15,[f'att_noisy_no_cyc_disc_{i}' for i in train_percents], csv_path, use_cycled_discriminators=False)
    

    # train_percents =[5,10,25,50]
    # vary_data(train_percents,15,[f'att_noisy_mask_{i}' for i in train_percents], csv_path,use_mask =True)

    # train_percents =[75]
    # vary_data(train_percents,15,[f'att_noisy_phase_{i}' for i in train_percents], csv_path,use_phase =True)

    # lambda_L1s = [2000]
    # vary_lambdaL1(lambda_L1s ,10,15,[f'pix_noisy_lambdL1_{i}' for i in lambda_L1s], csv_path)
    #run_eval()
    # codecs = ['g726','gsm','ogg','g723_1']
    # use_codecs(codecs , 10, 15, [f'pix_{c}' for c in codecs], csv_path)
    
