import torch
from cyclegan_model import ResNetGenerator, PatchGANDiscriminator
from train_cyclegan import CycleGANTrainer

def test_inference():
    print("Testing Model Inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = ResNetGenerator(n_blocks=6).to(device)
    D = PatchGANDiscriminator().to(device)
    
    dummy_input = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        out_G = G(dummy_input)
        out_D = D(dummy_input)
    
    print(f"G output shape: {out_G.shape}")
    print(f"D output shape: {out_D.shape}")
    assert out_G.shape == (1, 3, 128, 128)
    assert out_D.shape == (1, 1, 30, 30) # PatchGAN output for 128x128
    print("Inference Test Passed!")

def test_training_step():
    print("\nTesting Training Step...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = CycleGANTrainer(device)
    
    real_A = torch.randn(1, 3, 128, 128).to(device)
    real_B = torch.randn(1, 3, 128, 128).to(device)
    
    losses = trainer.train_step(real_A, real_B)
    print(f"Losses: {losses}")
    assert losses['loss_G'] > 0
    print("Training Step Test Passed!")

if __name__ == "__main__":
    test_inference()
    test_training_step()
