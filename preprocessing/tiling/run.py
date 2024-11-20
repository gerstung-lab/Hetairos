import subprocess
import time
import os
import numpy as np


dataset = 'dataset_name'
slide_dir = f'/path/to/slide/images/{dataset}' # Change this to the path of the slide images
save_dir = f'/save/path/{dataset}/20x_256' # Change this to the path where you want to save the tiles/masks/stitches
os.makedirs(save_dir, exist_ok=True)
os.makedirs('./log_file/', exist_ok=True)

try:
    existing_files = os.listdir(os.path.join(save_dir, 'tiles'))  # Check if the slide has already been processed
except FileNotFoundError:
    existing_files = []

formats = ['.svs', '.ndpi', '.scn']  # Add more formats if needed
slide_list = []
for root, dirs, files in os.walk(slide_dir):
    for file in files:
        if any(file.endswith(fmt) for fmt in formats):
            if os.path.splitext(file)[0] not in existing_files:
                slide_list.append(os.path.join(root, file))

sub_num = 5  # Number of slides to process in one job
job_num = int(np.ceil(len(slide_list)/sub_num)) # Number of jobs to submit
# tiling_params = {'patch_size': 256, 'step_size': 256, 'mag': 20} # Parameters for tiling, change if needed

for i in range(job_num):
    start = i*sub_num
    end = (i+1)*sub_num if (i+1)*sub_num < len(slide_list) else len(slide_list)
    slide_sub_list = slide_list[start:end]
    # slide_sub_list = '|'.join(slide_sub_list)
    list_loc_tmp = f'./{dataset}_list_{i}.txt'
    with open(list_loc_tmp, 'w') as f: # record the slides to be processed in a txt file
        for item in slide_sub_list:
            f.write("%s\n" % item)
    
    cmd = f'python -W ignore main_create_tiles.py --index {i} --source_list {list_loc_tmp} --save_dir {save_dir} --patch_size 256 --step_size 256 --mag 20'
    bsub_cmd = f'bsub -R "rusage[mem=30G]" -J {dataset}_{i} -q long -o ./log_{i}.out -e ./log_{i}.err {cmd}'
    try:
        subprocess.run(bsub_cmd, shell=True)
        time.sleep(1)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while submitting job {i}: {e}")
