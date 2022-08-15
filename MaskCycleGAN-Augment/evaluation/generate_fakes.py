from args.cycle_GAN_eval_arg_parser import CycleGANEvalArgParser
import os
import shutil
from data_preprocessing.preprocess_vcc2018 import preprocess_dataset
from mask_cyclegan_vc.test import MaskCycleGANVCTesting
import librosa
import soundfile as sf
# test comment
def main(args):

    source_id,source_path_end = (args.speaker_A_id,'src.wav') if args.model_name == 'generator_A2B' else (args.speaker_B_id,'A.wav')
    target_id,target_path_end = (args.speaker_A_id,'src.wav') if args.model_name == 'generator_B2A' else (args.speaker_B_id,'A.wav')

    # Creating temporary cache for data

    source_orig_data_path = os.path.join(args.eval_cache,'orig',source_id)
    target_orig_data_path = os.path.join(args.eval_cache,'orig',target_id)
    source_processed_data_path = os.path.join(args.eval_cache,'processed',source_id)
    target_processed_data_path = os.path.join(args.eval_cache,'processed',target_id)
    source_agg_processed_path = os.path.join(args.eval_cache,'agg',source_id)
    target_agg_processed_path = os.path.join(args.eval_cache,'agg',target_id)

    os.makedirs(os.path.join(args.eval_cache,'converted_audio','real'),exist_ok=True)

    #Preprocess all clips for aggregate statistics
    if not os.path.exists(os.path.join(args.eval_cache,'agg')):
        for speaker_id in [source_id,target_id]:
            preprocess_dataset(data_path=os.path.join(args.data_directory,speaker_id), speaker_id=speaker_id,
                            cache_folder=os.path.join(args.eval_cache,'agg'))
            os.remove(os.path.join(args.eval_cache,'agg',speaker_id,f'{speaker_id}_normalized.pickle'))

    count = 0
    total = len(os.listdir(os.path.join(args.data_directory,source_id)))

    for source_file in os.listdir(os.path.join(args.data_directory,source_id)):

        ## TODO : Generalize filename

        target_file = source_file[:-(len(source_path_end))]+target_path_end

        os.makedirs(source_orig_data_path)
        os.makedirs(target_orig_data_path)

        # Verifying format

        if source_file[-4:]!='.wav':
            print(f'Invalid Format. Skipping {source_file}')
            continue

        #Copy Data

        shutil.copyfile(os.path.join(args.data_directory,source_id,source_file),os.path.join(source_orig_data_path,source_file))
        shutil.copyfile(os.path.join(args.data_directory,target_id,target_file),os.path.join(target_orig_data_path,target_file))

        #Preprocess Data

        for speaker_id in [source_id,target_id]:
            stat = preprocess_dataset(data_path=os.path.join(args.eval_cache,'orig',speaker_id),
                                    speaker_id=speaker_id,cache_folder=os.path.join(args.eval_cache,'processed'),eval=True)
            if stat:
                os.remove(os.path.join(args.eval_cache,'processed',speaker_id,f'{speaker_id}_norm_stat.npz')) #Removing individual stats
                shutil.copyfile(os.path.join(args.eval_cache,'agg',speaker_id,f'{speaker_id}_norm_stat.npz'),\
                                        os.path.join(args.eval_cache,'processed',speaker_id,f'{speaker_id}_norm_stat.npz')) #Copying aggregated stats
            else:
                break
        
        if not stat:
            shutil.rmtree(source_orig_data_path)
            shutil.rmtree(target_orig_data_path)
            continue


        
        # Run inference
        args.eval = True
        args.eval_save_dir = os.path.join(args.eval_cache,'converted_audio','generated')
        args.filename = source_file[:-(len(source_path_end)+1)]+'.wav'
        args.preprocessed_data_dir = os.path.join(args.eval_cache,'processed')
        tester = MaskCycleGANVCTesting(args)
        tester.test()

        #Copy original target file to Real folder with given sample_rate
        real , sr = librosa.load(os.path.join(target_orig_data_path,target_file))
        sf.write(os.path.join(args.eval_cache,'converted_audio','real',source_file[:-(len(source_path_end)+1)]+'.wav'), real, args.sample_rate, 'PCM_16')
       
        
        #Deleting Processed and Orig Directories
        shutil.rmtree(source_orig_data_path)
        shutil.rmtree(target_orig_data_path)
        shutil.rmtree(source_processed_data_path)
        shutil.rmtree(target_processed_data_path)

        count+=1

        print(f"{'-'*10} Processed {count}/{total} {'-'*10}")
        

if __name__ == "__main__":
    parser = CycleGANEvalArgParser()
    args = parser.parse_args()
    main(args)