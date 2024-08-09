import pandas as pd
import json
import yaml
import os

class DataInfoLoader:
    def __init__(self, dataset_name, config):
        self.config = config
        self.dataset_name = dataset_name
        self.factor = 1.0  # Manually change the factor here as needed

        # Load ground truth data from the specified file
        gt_file_path = config[dataset_name]['gt_file_path']
        with open(gt_file_path, 'r') as f:
            self.gt_data = json.load(f)

        self.img_num = len(self.gt_data)
        self.IQA_results_path = config[dataset_name]['IQA_results_path']

    def get_qs_std(self):
        """Standard ground truth quality score of all images

        Returns:
        - pandas.Series storing the GT for all images
        """
        return pd.Series([item['distortion_level'][0] * self.factor if isinstance(item['distortion_level'], list) else item['distortion_level'] * self.factor for item in self.gt_data])

    def get_img_name(self):
        return pd.Series([item['image'] for item in self.gt_data])

    def get_img_set(self):
        return pd.Series([self.dataset_name] * self.img_num)

    def get_img_path(self):
        """Get the image path for all images"""
        return pd.Series([os.path.join(self.config[self.dataset_name]['root'], 'contrast_change', item['image']) for item in self.gt_data])

if __name__ == '__main__':
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    dataset_name = 'SOC'
    dil = DataInfoLoader(dataset_name, config)
    img_name = dil.get_img_name()
    print(img_name[2])
