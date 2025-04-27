import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# ========== Feature Extraction Functions ==========

def extract_dct_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    return cv2.resize(dct, (224, 224)).astype(np.float32)

def extract_noise_residue(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - denoised
    return cv2.resize(noise, (224, 224)).astype(np.float32)

def extract_compression_artifacts(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    diff = image.astype(np.float32) - decimg.astype(np.float32)
    return cv2.resize(diff, (224, 224)).astype(np.float32)

# ========== Custom Dataset Class ==========

class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))

        dct_feat = extract_dct_features(image)
        noise_feat = extract_noise_residue(image)
        comp_feat = extract_compression_artifacts(image)

        image_tensor = self.transform(image)
        dct_tensor = torch.tensor(dct_feat).unsqueeze(0)
        noise_tensor = torch.tensor(noise_feat).unsqueeze(0)
        comp_tensor = torch.tensor(comp_feat.transpose(2, 0, 1))

        return image_tensor, dct_tensor, noise_tensor, comp_tensor, torch.tensor(label, dtype=torch.long)

# ========== Deepfake Detection Model ==========

class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        self.base_model = efficientnet_b4(weights=weights)
        self.base_model.classifier = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(1792 + 224*224 + 224*224 + 3*224*224, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, img, dct, noise, comp):
        x = self.base_model(img)
        dct = dct.view(dct.size(0), -1)
        noise = noise.view(noise.size(0), -1)
        comp = comp.reshape(comp.size(0), -1)

        fused = torch.cat((x, dct, noise, comp), dim=1)
        out = self.fc(fused)
        return out

# ========== Training Function ==========

def train_model(dataset_path, epochs=10, batch_size=8, lr=1e-4):
    real_images = glob(os.path.join(dataset_path, 'real', '*'))
    fake_images = glob(os.path.join(dataset_path, 'fake', '*'))
    all_images = real_images + fake_images
    labels = [0] * len(real_images) + [1] * len(fake_images)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_images, labels, test_size=0.2, stratify=labels, random_state=42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = DeepfakeDataset(train_paths, train_labels, transform)
    val_dataset = DeepfakeDataset(val_paths, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for img, dct, noise, comp, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            img, dct, noise, comp, label = img.to(device), dct.to(device), noise.to(device), comp.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(img, dct, noise, comp)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for img, dct, noise, comp, label in val_loader:
                img, dct, noise, comp, label = img.to(device), dct.to(device), noise.to(device), comp.to(device), label.to(device)
                output = model(img, dct, noise, comp)
                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%\n")
        torch.save(model.state_dict(), "model.pth")


# ========== Run Training ==========

if __name__ == "__main__":
    train_model("/Users/zeeshan/Downloads/Final")