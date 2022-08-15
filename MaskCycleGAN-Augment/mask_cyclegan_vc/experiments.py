import os
import pandas as pd
import shutil
import sys
from metrics.lsd import main as lsd
from metrics.mssl import main as mssl

def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)

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
        run(f'python -W ignore::UserWarning -m mask_cyclegan_vc.test --name {name} --split val --save_dir {results_dir} --gpu_ids 0 --speaker_A_id clean --speaker_B_id noisy --load_epoch {epoch} --ckpt_dir /content/drive/MyDrive/APSIPA/Results/{name}/ckpts --model_name generator_A2B')
        avg_lsd,std_lsd = lsd(os.path.join(data_cache,'noisy','val'),os.path.join(results_dir,name,'audios','fake_B'),use_gender=False)
        avg_mssl,std_mssl = mssl(os.path.join(data_cache,'noisy','val'),os.path.join(results_dir,name,'audios','fake_B'),use_gender=False)

        shutil.rmtree(os.path.join(results_dir,name,'audios','fake_B'))

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


def apsipa_exp(names,csv_path,sources, data_cache='/content/MaskCycleGAN-Augment/data_cache',results_dir='/content/MaskCycleGAN-Augment/results', epochs = [25,50,75,100,125,150]):
     for name, source in zip(names,sources):
        print('#'*25)
        print(f'Training {name} with Data from {source}')
        shutil.copytree(os.path.join('/content/drive/MyDrive/APSIPA/Data_Sources',source),data_cache)
        run(f'python -W ignore::UserWarning -m mask_cyclegan_vc.train --name {name} --seed 0 --save_dir /content/drive/MyDrive/APSIPA/Results --speaker_A_id clean --speaker_B_id noisy --epochs_per_save 25 --epochs_per_plot 10 --num_epochs 150 --batch_size 8 --decay_after 1e4 --sample_rate 8000 --num_frames 64 --max_mask_len 50 --gpu_ids 0 --generator_lr 5e-4 --discriminator_lr 5e-4 --preprocess resize')
        
        info = validate(name, epochs, data_cache, results_dir)
        for metric in ['min_lsd_epoch','min_mssl_epoch']:
            epoch = info[metric]
            run(f'python -W ignore::UserWarning -m mask_cyclegan_vc.test --name {name} --split test --save_dir {results_dir} --gpu_ids 0 --speaker_A_id clean --speaker_B_id noisy --load_epoch {epoch} --ckpt_dir /content/drive/MyDrive/APSIPA/Results/{name}/ckpts --model_name generator_A2B')
            info[f'avg_test_lsd_{metric}'],info[f'std_test_lsd_{metric}'] = lsd(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'audios','fake_B'),use_gender=False)
            info[f'avg_test_mssl_{metric}'],info[f'std_test_mssl_{metric}'] = mssl(os.path.join(data_cache,'noisy','test'),os.path.join(results_dir,name,'audios','fake_B'),use_gender=False)

            shutil.rmtree(os.path.join(results_dir,name,'audios','fake_B'))

            if info['min_mssl_epoch'] == info['min_lsd_epoch']:
                info['avg_test_lsd_min_mssl_epoch'] = info['avg_test_lsd_min_lsd_epoch']
                info['std_test_lsd_min_mssl_epoch'] = info['std_test_lsd_min_lsd_epoch']
                info['avg_test_mssl_min_mssl_epoch'] = info['avg_test_mssl_min_lsd_epoch']
                info['std_test_mssl_min_mssl_epoch'] = info['std_test_mssl_min_lsd_epoch']
                break

        info['name'] = name
        info['comment'] = f'MaskCycleGAN trained for 150 epochs, WITHOUT phase on {source} dataset.'
        log(csv_path, info)

        shutil.rmtree(data_cache)
        print(f'Finished experiment with {name}')
        print('#'*25)

if __name__ == '__main__':
    csv_path = '/content/drive/MyDrive/APSIPA/Results/logs.csv'
    if not os.path.exists(csv_path):
        cols=['name','comment','min_lsd_epoch','min_mssl_epoch','avg_test_lsd_min_lsd_epoch','avg_test_mssl_min_lsd_epoch',	'avg_test_lsd_min_mssl_epoch','avg_test_mssl_min_mssl_epoch','min_val_lsd','min_val_mssl','std_test_lsd_min_lsd_epoch','std_test_mssl_min_lsd_epoch','std_test_lsd_min_mssl_epoch',	'std_test_mssl_min_mssl_epoch','avg_val_lsd','avg_val_mssl','std_val_lsd','std_val_mssl']
        df=pd.DataFrame(columns=cols)
        df.to_csv(csv_path,index=False)
    

    sources = ['Non-Parallel/TIMIT_Cabin','Non-Parallel/Codec2']
    apsipa_exp([f'MaskCycleGAN_{i}' for i in ['np_cabin','np_cd2']],csv_path,sources)
    