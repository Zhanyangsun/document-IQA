from PIL import Image
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torch.autograd import Variable as V
from scipy.signal import convolve2d
import cv2
import os


class CNNDIQAnet(nn.Module):
    def __init__(self):
        super(CNNDIQAnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, 5)
        self.pool1 = nn.MaxPool2d(4)
        self.conv2 = nn.Conv2d(40, 80, 5)
        self.fc1 = nn.Linear(160, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)  # Changed to output 10-dimensional vector

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x1 = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        x2 = -F.max_pool2d(-x, (x.size(-2), x.size(-1)))
        x = torch.cat((x1, x2), 1)
        x = x.squeeze(3).squeeze(2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def localNormalization(patch, P=3, Q=3, C=1):
    """Apply local normalization to an image patch."""
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = (patch - patch_mean) / patch_std
    return patch_ln


def judgeAllOnesOrAllZeros(patch):
    flag1 = np.all(patch == 255)
    flag2 = np.all(patch == 0)
    return flag1 or flag2


def patchSifting(im, patch_size=48, stride=48):
    """Extract informative patches from an image."""
    img = np.array(im).copy()
    im1 = localNormalization(img)
    im1 = Image.fromarray(im1)
    _, im2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    w, h = im1.size
    patches = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = im2[i:i + patch_size, j:j + patch_size]
            if not judgeAllOnesOrAllZeros(patch):
                patch = to_tensor(im1.crop((j, i, j + patch_size, i + patch_size)))
                patch = patch.float().unsqueeze(0)
                patches = patches + (patch,)
    return patches


class Solver:
    def __init__(self):
        # Pre-trained model path
        self.model_path = './checkpoints/CNNDIQA-SOC-EXPop-lr=0.0005.pth'
        # Initialize the model
        self.model = CNNDIQAnet()

    def quality_assessment(self, img_path):
        # Load the pre-trained model
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        im = Image.open(img_path).convert('L')
        patches = torch.stack(patchSifting(im))
        qs = self.model(patches)
        qs = qs.mean(dim=0).detach().cpu().numpy()  # Detach the tensor before converting to NumPy
        return qs


if __name__ == '__main__':
    base_path = './dataset'
    folder_name = 'optical_problems'  # Specify the name for the folder containing images
    output_file = 'optical_problems_quality_scores.txt'
    solver = Solver()

    # Process all images in the specified folder
    with open(output_file, 'w') as f:
        dir_path = os.path.join(base_path, folder_name)
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            if img_path.endswith('.png'):
                qs = solver.quality_assessment(img_path)
                f.write(f'{img_name}: {qs.tolist()}\n')

    print(f'Quality scores for images in {folder_name} saved to {output_file}')
