import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from PIL import Image
import os

# Define CNNDIQAnet model
class CNNDIQAnet(nn.Module):
    def __init__(self):
        super(CNNDIQAnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, 5)
        self.pool1 = nn.MaxPool2d(4)
        self.conv2 = nn.Conv2d(40, 80, 5)
        self.fc1 = nn.Linear(160, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

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

# MLP Model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def judgeAllOnesOrAllZeros(patch):
    return np.all(patch == 255) or np.all(patch == 0)

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
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)  # Ensure shape [1, patch_size, patch_size]
                patches.append(patch_tensor)
    return patches

def quality_assessment(model, patches):
    model.eval()
    with torch.no_grad():
        qs = model(patches)
        qs_mean = qs.mean(dim=0).detach().cpu().numpy()  # Compute mean for each dimension
        return qs_mean

def evaluate_image(img_path, model_path_1, model_path_2, mlp_model_path, device, factors):
    # Load models
    model_1 = CNNDIQAnet()
    model_1.load_state_dict(torch.load(model_path_1, map_location=device))
    model_1.to(device)

    model_2 = CNNDIQAnet()
    model_2.load_state_dict(torch.load(model_path_2, map_location=device))
    model_2.to(device)

    mlp_model = MLP()
    mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device))
    mlp_model.to(device)
    mlp_model.eval()

    # Process image
    im = Image.open(img_path).convert('L')
    patches = torch.stack(patchSifting(im)).to(device)

    # Get quality scores from both models
    y_pred_1 = quality_assessment(model_1, patches)
    y_pred_2 = quality_assessment(model_2, patches)

    # Combine scores and apply factors
    combined_scores = torch.tensor([y_pred_1, y_pred_2], dtype=torch.float32).to(device).view(1, -1) * torch.tensor(factors, dtype=torch.float32).to(device)

    # Get final quality score from MLP model and adjust with factors
    final_score = mlp_model(combined_scores).detach().cpu().numpy() / np.array(factors)

    return final_score

if __name__ == '__main__':
    img_path = './dataset/1.png'  # Replace with your image path
    model_path_1 = './checkpoints/CNNDIQA-SOC-EXPbinary_threshold-lr=0.001.pth'  # Replace with the path to your first CNN model
    model_path_2 = './checkpoints/CNNDIQA-SOC-EXPgaussian_blur-lr=0.0005.pth'  # Replace with the path to your second CNN model
    mlp_model_path = './checkpoints/mlp_model.pth'  # Replace with the path to your MLP model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factors = [1.0, 10.0]  # Specify the factors to apply to each value

    score = evaluate_image(img_path, model_path_1, model_path_2, mlp_model_path, device, factors)
    print(f'Quality score for the image: {score}')
