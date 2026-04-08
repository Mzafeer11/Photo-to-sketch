import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_utils import UnpairedDataset
from train_cyclegan import CycleGANTrainer
import os

# Configuration
BATCH_SIZE = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = '/kaggle/working/checkpoints'
DATA_ROOT_A = "sketchy/sketchy/sketches"
DATA_ROOT_B = "sketchy/sketchy/photos"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_cyclegan(epochs=10):
    # Transforms (Updated to 128x128 for Kaggle T4 performance)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset
    dataset = UnpairedDataset(DATA_ROOT_A, DATA_ROOT_B, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Trainer
    trainer = CycleGANTrainer(DEVICE)

    # Resume capability
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_cyclegan.pth')
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        trainer.G_AB.load_state_dict(checkpoint['G_AB'])
        trainer.G_BA.load_state_dict(checkpoint['G_BA'])
        trainer.D_A.load_state_dict(checkpoint['D_A'])
        trainer.D_B.load_state_dict(checkpoint['D_B'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training Loop
    for epoch in range(start_epoch, epochs):
        for i, batch in enumerate(loader):
            real_A = batch['A'].to(DEVICE)
            real_B = batch['B'].to(DEVICE)
            
            losses = trainer.train_step(real_A, real_B)
            
            if i % 100 == 0:
                print(f"Epoch {epoch} Batch {i} | G: {losses['loss_G']:.4f} D: {losses['loss_D']:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'G_AB': trainer.G_AB.state_dict(),
            'G_BA': trainer.G_BA.state_dict(),
            'D_A': trainer.D_A.state_dict(),
            'D_B': trainer.D_B.state_dict(),
        }, checkpoint_path)

if __name__ == "__main__":
    train_cyclegan()
