import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np
from DataInfoLoader import DataInfoLoader

def default_loader(path):
    return Image.open(path).convert('L')

def patchSifting(im, patch_size=250, stride=250):
    img = np.array(im).copy()

    if not img.flags.writeable:
        img = np.copy(img)

    h, w = img.shape
    patches = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img[i:i + patch_size, j:j + patch_size]
            if not judgeAllOnesOrAllZeros(patch):
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)  # Ensure shape [1, 48, 48]
                patches.append(patch_tensor)
    return patches

def judgeAllOnesOrAllZeros(patch):
    return np.all(patch == 255) or np.all(patch == 0)

class DIQADataset(Dataset):
    def __init__(self, dataset_name, config, data_index, status='train', loader=default_loader):
        self.loader = loader
        dil = DataInfoLoader(dataset_name, config)
        img_name = dil.get_img_name()
        img_path = dil.get_img_path()
        qs_std = dil.get_qs_std()

        self.index = data_index
        print(f"# {status.capitalize()} Images: {len(self.index)}")
        print('Index:')
        print(self.index)

        self.patches = []
        self.label = []

        for idx in self.index:
            im = self.loader(img_path[idx])
            patches = patchSifting(im)
            if status == 'train':
                self.patches.extend(patches)
                self.label.extend([torch.tensor(qs_std[idx], dtype=torch.float32).view(1)] * len(patches))
            else:
                self.patches.extend(patches)
                self.label.extend([torch.tensor(qs_std[idx], dtype=torch.float32).view(1)] * len(patches))

        if status == 'train' or status == 'val' or status == 'test':
            print(f"Total Patches: {len(self.patches)}")
            print(f"Total Labels: {len(self.label)}")
            if len(self.patches) > 0:
                print(f"Sample Patch Shape: {self.patches[0].shape}")
            if len(self.label) > 0:
                print(f"Sample Label Shape: {self.label[0].shape}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch, label = self.patches[idx], self.label[idx]
        # print(f"__getitem__: patch shape: {patch.shape}, label shape: {label.shape}")  # Debug statement
        return patch, label
