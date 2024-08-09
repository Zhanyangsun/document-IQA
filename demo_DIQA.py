from PIL import Image
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torch.autograd import Variable as V
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
        self.fc3 = nn.Linear(1024, 2)  # Changed to output 2-dimensional vector

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


def judgeAllOnesOrAllZeros(patch):
    return np.all(patch == 255) or np.all(patch == 0)


def patchSifting(im, patch_size=350, stride=350):
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


class Solver:
    def __init__(self):
        # Pre-trained model path
        self.model_path = './checkpoints/CNNDIQA-SOC-EXPbinary_threshold_gaussian_blur-lr=0.0005.pth'
        # Initialize the model
        self.model = CNNDIQAnet()
        self.factors = np.array([1.0,10.0])  # Scaling factors for each dimension

    def quality_assessment(self, img_path):
        # Load the pre-trained model
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        im = Image.open(img_path).convert('L')
        patches = torch.stack(patchSifting(im))
        qs = self.model(patches)
        qs_mean = qs.mean(dim=0).detach().cpu().numpy()  # Compute mean for each dimension
        qs_scaled = qs_mean / self.factors  # Apply inverse scaling
        return qs_scaled


if __name__ == '__main__':
    base_path = './dataset1'
    folder_name = 'binary_threshold_gaussian_blur'  # Specify the name for the folder containing images
    output_file = 'binary_threshold_gaussian_blur_quality_scores.txt'
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
