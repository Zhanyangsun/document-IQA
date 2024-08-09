import os
import json
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torchvision.transforms.functional import to_grayscale
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

# CNN Model Placeholder
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


class DataInfoLoader:
    def __init__(self, dataset_name, config):
        self.config = config
        self.dataset_name = dataset_name

        gt_file_path = config[dataset_name]['gt_file_path']
        with open(gt_file_path, 'r') as f:
            self.gt_data = json.load(f)

        self.img_num = len(self.gt_data)

    def get_img_name(self):
        return [item['image'] for item in self.gt_data]

    def get_img_path(self):
        return [os.path.join(self.config[self.dataset_name]['root'], 'binary_threshold_gaussian_blur', item['image']) for item in self.gt_data]

    def get_qs_std(self):
        return [
            [item['distortion_level'][0], item['distortion_level'][1]]
            for item in self.gt_data
        ]

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
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)
                patches.append(patch_tensor)
    return patches

def quality_assessment(model, patches):
    model.eval()
    with torch.no_grad():
        qs = model(patches)
        qs_mean = qs.mean(dim=0).detach().cpu().numpy()
        return qs_mean

def evaluate_images(dataset_name, config, model_path_1, model_path_2, output_json_path, factors, disable_gpu=False):
    device = torch.device("cuda" if torch.cuda.is_available() and not disable_gpu else "cpu")

    model_1 = CNNDIQAnet()
    model_1.load_state_dict(torch.load(model_path_1, map_location=device))
    model_1.to(device)

    model_2 = CNNDIQAnet()
    model_2.load_state_dict(torch.load(model_path_2, map_location=device))
    model_2.to(device)

    data_loader = DataInfoLoader(dataset_name, config)
    img_paths = data_loader.get_img_path()
    img_names = data_loader.get_img_name()

    scores_dict = {}
    for img_name, img_path in zip(img_names, img_paths):
        im = Image.open(img_path).convert('L')
        patches = torch.stack(patchSifting(im)).to(device)
        y_pred_1 = quality_assessment(model_1, patches)
        y_pred_2 = quality_assessment(model_2, patches)
        scores_dict[img_name] = [y_pred_1.tolist(), y_pred_2.tolist()]

    with open(output_json_path, 'w') as f:
        json.dump(scores_dict, f)

    print(f'Scores saved to {output_json_path}')

class ScoreDataset(Dataset):
    def __init__(self, score_file, config, dataset_name, factors):
        self.factors = torch.tensor(factors, dtype=torch.float32)
        with open(score_file, 'r') as f:
            self.scores = json.load(f)

        datainfo = DataInfoLoader(dataset_name, config)
        qs_std = datainfo.get_qs_std()
        self.data = [(torch.tensor(self.scores[img], dtype=torch.float32).view(-1) * self.factors, torch.tensor(qs, dtype=torch.float32).view(-1) * self.factors) for img, qs in zip(self.scores.keys(), qs_std)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_mlp(score_file, config, dataset_name, mlp_model_save_path, batch_size=128, epochs=10000, lr=0.005, weight_decay=0.0, factors=[1.0, 1.0], disable_gpu=False):
    print("Training started...")
    device = torch.device("cuda" if torch.cuda.is_available() and not disable_gpu else "cpu")
    print(f"Using device: {device}")

    dataset = ScoreDataset(score_file, config, dataset_name, factors)
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    mlp_model = MLP().to(device)
    optimizer = Adam(mlp_model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')

    # Train the MLP model
    print("Training started")
    mlp_model.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        for data in train_loader:
            optimizer.zero_grad()
            combined_scores, labels = data
            combined_scores, labels = combined_scores.to(device), labels.to(device)
            y_pred = mlp_model(combined_scores)
            y_pred = y_pred.view_as(labels)  # Ensure y_pred has the same shape as labels
            loss = F.mse_loss(y_pred, labels)
            loss.backward()
            optimizer.step()

        # Validation
        if (epoch + 1) % 1000 == 0:
            mlp_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in val_loader:
                    combined_scores, labels = data
                    combined_scores, labels = combined_scores.to(device), labels.to(device)
                    val_pred = mlp_model(combined_scores)
                    val_pred = val_pred.view_as(labels)  # Ensure val_pred has the same shape as labels
                    val_loss += F.mse_loss(val_pred, labels).item()
                val_loss /= len(val_loader)
                print(f"Validation Loss: {val_loss:.4f}")

                # Save the best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(mlp_model.state_dict(), mlp_model_save_path)

            mlp_model.train()

    print("Training completed.")

    # Testing
    mlp_model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for data in test_loader:
            combined_scores, labels = data
            combined_scores, labels = combined_scores.to(device), labels.to(device)
            test_pred = mlp_model(combined_scores)
            test_pred = test_pred.view_as(labels)  # Ensure test_pred has the same shape as labels
            test_loss += F.mse_loss(test_pred, labels).item()
        test_loss /= len(test_loader)
        print(f"Final Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    batch_size = 8
    epochs = 10000
    lr = 0.005
    weight_decay = 0.0
    config_path = 'config_ensemble.yaml'
    dataset_name = 'SOC'
    model_path_1 = './checkpoints/CNNDIQA-SOC-EXPbinary_threshold-lr=0.001.pth'
    model_path_2 = './checkpoints/CNNDIQA-SOC-EXPgaussian_blur-lr=0.0005.pth'
    output_json_path = './checkpoints/image_scores.json'
    mlp_model_save_path = './checkpoints/mlp_model.pth'
    disable_gpu = False
    factors = [1.0, 10.0]

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Evaluate and save scores
    evaluate_images(dataset_name, config, model_path_1, model_path_2, output_json_path, factors, disable_gpu)
    # Train MLP with the factors applied
    train_mlp(output_json_path, config, dataset_name, mlp_model_save_path, batch_size, epochs, lr, weight_decay, factors, disable_gpu)
