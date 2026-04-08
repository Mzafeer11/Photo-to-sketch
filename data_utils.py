import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

class UnpairedDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None, split_file_A=None, split_file_B=None, mode='train'):
        self.root_A = root_A # Sketch Root
        self.root_B = root_B # Photo Root
        self.transform = transform
        self.mode = mode
        
        # Load all available files
        all_A = self.get_all_files(root_A)
        all_B = self.get_all_files(root_B)
        
        # Load test sets if provided
        test_A = self.load_lines(split_file_A) if split_file_A else []
        test_B = self.load_lines(split_file_B) if split_file_B else []
        
        if mode == 'test':
            self.files_A = [f for f in all_A if any(ts in f for ts in test_A)]
            self.files_B = [f for f in all_B if any(tb in f for tb in test_B)]
        else:
            # Train mode: exclude test files
            self.files_A = [f for f in all_A if not any(ts in f for ts in test_A)]
            self.files_B = [f for f in all_B if not any(tb in f for tb in test_B)]
        
        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)

    def load_lines(self, split_file):
        if not os.path.exists(split_file): return []
        with open(split_file, 'r') as f:
            return [l.strip().replace('.jpg', '').replace('.png', '') for l in f if l.strip() and not l.startswith(('-', 'S'))]

    def get_all_files(self, directory):
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    files.append(os.path.join(root, filename))
        return files

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % self.len_A]).convert("RGB"))
        item_B = self.transform(Image.open(self.files_B[random.randint(0, self.len_B - 1)]).convert("RGB"))
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(self.len_A, self.len_B)

def get_transforms(img_size=128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

def get_dataloaders(root_A, root_B, batch_size=4, img_size=128):
    transform = get_transforms(img_size)
    dataset = UnpairedDataset(root_A, root_B, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader

# Example Kaggle Paths (to be used in the notebook)
# root_sketch = '/kaggle/input/tu-berlin/tu-berlin/images'
# root_photo = '/kaggle/input/sketchy-dataset/sketchy/photos'
