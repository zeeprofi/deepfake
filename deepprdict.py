import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from model import DeepfakeModel, extract_dct_features, extract_noise_residue, extract_compression_artifacts

# ========== Custom Dataset for Prediction ==========

class PredictDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_paths = [image_path]  # Now just a single image path
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))

        dct_feat = extract_dct_features(image)
        noise_feat = extract_noise_residue(image)
        comp_feat = extract_compression_artifacts(image)

        image_tensor = self.transform(image)
        dct_tensor = torch.tensor(dct_feat).unsqueeze(0)
        noise_tensor = torch.tensor(noise_feat).unsqueeze(0)
        comp_tensor = torch.tensor(comp_feat.transpose(2, 0, 1))

        return image_tensor, dct_tensor, noise_tensor, comp_tensor, img_path

# ========== Prediction Function ==========

def predict_images(model_path, image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = PredictDataset(image_path, transform)  # Pass the single image path
    dataloader = DataLoader(dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepfakeModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = {}
    with torch.no_grad():
        for img, dct, noise, comp, path in dataloader:
            img, dct, noise, comp = img.to(device), dct.to(device), noise.to(device), comp.to(device)

            # Get the model output (logits)
            output = model(img, dct, noise, comp)
            
            # Get predicted class (Real or Fake) and confidence score
            _, pred = torch.max(output, 1)
            confidence = torch.nn.functional.softmax(output, dim=1)[0][pred].item()
            
            # Get the individual logits for both Real (0) and Fake (1) classes
            real_logit, fake_logit = output[0].cpu().numpy()

            # Mapping prediction and label
            label = 'Real' if pred.item() == 1 else 'Fake'

            predictions[path[0]] = {
                'Prediction': label,
                'Confidence': confidence,
                'Real Logit': real_logit,
                'Fake Logit': fake_logit,
                'DCT Features': dct.cpu().numpy().tolist(),
                'Noise Features': noise.cpu().numpy().tolist(),
                'Compression Features': comp.cpu().numpy().tolist()
            }

            print(f"{os.path.basename(path[0])}: {label}")
            print(f" - Confidence: {confidence:.4f}")
            print(f" - Real Logit: {real_logit:.4f}, Fake Logit: {fake_logit:.4f}")
            print(f" - DCT Features: {dct.cpu().numpy().flatten()[:5]}...")  # Displaying the first 5 features
            print(f" - Noise Features: {noise.cpu().numpy().flatten()[:5]}...")
            print(f" - Compression Features: {comp.cpu().numpy().flatten()[:5]}...")

    return predictions

# ========== Run Prediction ==========

if __name__ == "__main__":
    model_checkpoint_path = "model.pth"
    test_image_path = input("Enter the path of the image you want to predict: ")
    predictions = predict_images(model_checkpoint_path, test_image_path)

    # Optionally, save predictions to a file for further analysis
    with open('predictions.txt', 'w') as f:
        for img_path, details in predictions.items():
            f.write(f"{img_path}:\n")
            for key, value in details.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")