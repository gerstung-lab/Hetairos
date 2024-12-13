import subprocess
import time
import os
import glob
import numpy as np


dataset = 'datasetname' 
tile_dir = f'/path/to/tile/images/{dataset}/20x_256/tiles'  # Change this to the path of the tiles
save_dir = f'/save/path/ProvGigaPath_256_1536_{dataset}' # Change this to the path where you want to save the embeddings
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, 'pt_files'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'h5_files'), exist_ok=True)

existing_files = os.listdir(os.path.join(save_dir, 'pt_files'))
existing_files = [item.split('.')[0]for item in existing_files]
slide_list_ = glob.glob(os.path.join(tile_dir, '*'))
slide_list = [item for item in slide_list_ if os.path.basename(item) not in existing_files]

sub_num = 50 # Number of slides to process in one job
job_num = int(np.ceil(len(slide_list)/sub_num)) # Number of jobs to submit
for i in range(job_num):
    start = i*sub_num
    end = (i+1)*sub_num if (i+1)*sub_num < len(slide_list) else len(slide_list)
    slide_sub_list = slide_list[start:end] # List of slides to process in this job
    list_loc_tmp = f'./{dataset}_list_{i}.txt'
    with open(list_loc_tmp, 'w') as f:
        for item in slide_sub_list:
            f.write("%s\n" % item)

    batchsize = 768
    cmd = f"python -W ignore feature_extraction/get_features.py --split '{list_loc_tmp}' --batchsize {batchsize} --feature_dir {save_dir}"
    bsub_cmd = f'bsub -gpu num=1:j_exclusive=yes:gmem=23.5G -R "rusage[mem=20G]" -L /bin/bash -q gpu -J {dataset}_{i} -o ./log_{i}.log -e ./log_{i}.err "source ~/.bashrc && {cmd}"'
    try:
        subprocess.run(bsub_cmd, shell=True)
        time.sleep(3)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while submitting job {i}: {e}")

