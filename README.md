# DIQA_CNN
## Data
The medical forms data are stored in data folder. All training data is generated based on these.

## Piepeline (ensemble learning)
### Generate training image (single noise)
1. In data_prep_single_noise.py, edit the noise types in main function and run the script, noise type is a list like ['binary_threshold', 'gaussian_blur', 'optical_problems', 'rotation', 'contrast_change', 'pixelate', 'jitter', 'mean_shift', 'salt_and_pepper']. This will generate folders with the corresponding noise name under the dataset folder along with the ground truth.
### Train model (single noise)
2. Edit gt_file_path in config.yaml to be the ground truth path of the noise you want to use
3. In DataInfoLoader.py >> DataInfoLoader class >> get_img_path function, the noise name is specified, change correspondingly.
4. In main.py, you can specify the learning rate, batch size etc. Don't foeget to specify the exp_id. 
5. Finally, run main.py. The model is saved under the checkpoint folder.
### Generate training image (two noise)
6. In data_prep.py, edit the noise types in main function and run the script, noise_type1 and noise_type2 should each be an element of ['binary_threshold', 'gaussian_blur', 'optical_problems', 'rotation', 'contrast_change', 'pixelate', 'jitter', 'mean_shift', 'salt_and_pepper']. This will generate a folder with the corresponding noise name under the dataset folder along with the ground truth. 
7. Edit gt_file_path in config_ensemble.yaml to be the ground truth path of the noise you want to use
8. In ensemble.py >> DataInfoLoader class >> get_img_path function, the noise name is specified, change correspondingly (name of the folder that stores the noisy images)
9. In ensemble.py >> main method, you need to specify model_path_1, model_path_2
10. Run ensemble.py, you can adjust any parameters you want. The factors parameter is used to scale the noise if you want.

** Other files are remnant from different testing endeavors, and do not influence the running of this pipeline

