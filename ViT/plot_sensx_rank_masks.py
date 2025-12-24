import os
from PIL import Image
import sys
import pickle
from tqdm import tqdm
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append('../')

import sensx_landscapes_helper as slh

raw_input_dir = "data/"

runs = 10

for category in ['smiling', 'eyeglasses']:

    #########################################
    
    if category == 'smiling':
        model_path = "./vit-Smiling-model-final"
        results_dir = 'smiling_analysis_results'
    elif category == 'eyeglasses':
        model_path = "./vit-Eyeglasses-model-final"
        results_dir = 'eyeglasses_analysis_results'
    
    
    #########################################
    ## LOAD SENSITIVIY MAP
    
    
    sensitivity_map = None
    
    found = 0
    skip = 0

    all_raw_inputs = None
    all_sensitivity_maps = None
    
    for rr in tqdm(range(runs)):

        if rr < 10:
            fname = f'{results_dir}/run_0{rr}/results_config_{category}_run_0{rr}.pkl'
        else:
            fname = f'{results_dir}/run_{rr}/results_config_{category}_run_{rr}.pkl'

        try:
    
            dbfile = open(fname, 'rb')
            data = pickle.load(dbfile)
            dbfile.close()

            print(data['delta_star'])


            img_paths = data['filenames']
            image_inputs = data['input']

            if all_raw_inputs is None:

                all_raw_inputs = []

                for fname in img_paths:

                    raw_image = Image.open(f'data/{fname}').convert("RGB")
                    raw_image = raw_image.resize((224, 224))
                    image_arr = np.array(raw_image)

                    all_raw_inputs.append(image_arr)

                all_raw_inputs = np.array(all_raw_inputs)


            if all_sensitivity_maps is None:

                all_sensitivity_maps = data['sensitivity_map'].numpy()
                found = 1

            else:

                all_sensitivity_maps += data['sensitivity_map'].numpy()
                found += 1

        except:

            skip += 1


    print(f'found: {found}, skip: {skip}')
    
    all_sensitivity_maps /= found

    for img_name, image_arr, sensitivity_map in zip(img_paths, all_raw_inputs, all_sensitivity_maps):

    
        ranks = rankdata(-sensitivity_map, method='average').reshape(sensitivity_map.shape)

        
        ranks = np.transpose(ranks, (1, 2, 0))
        
        
        #########################################
        
        top_n = [1000, 2500, 5000, 7500]
        
        fig, axs = plt.subplots(1, 5, figsize=(18,6))
        
        axs[0].imshow(image_arr)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        
        for ii, tt in enumerate(top_n):
        
            top_mask = ranks > tt
        
            image_copy = image_arr.copy()
        
            image_copy[top_mask] = 255
        
            axs[ii+1].imshow(image_copy)
        
            axs[ii+1].set_title(image_copy)
        
            axs[ii+1].set_xticks([])
            axs[ii+1].set_yticks([])
        
            axs[ii+1].set_title(f'Top {tt} SensX features\n{category}-ViT', fontsize=16)
        
        
        plt.tight_layout()
        plt.savefig(f'sensx_masks/sensx_masks_{category}_{img_name}.jpg', dpi=300)
        
        plt.cla()
        plt.close()




