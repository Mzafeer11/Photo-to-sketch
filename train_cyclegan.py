import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import itertools
from cyclegan_model import ResNetGenerator, PatchGANDiscriminator, init_weights

class CycleGANTrainer:
    def __init__(self, device, lr=0.0002, beta1=0.5, beta2=0.999, lambda_cycle=10.0, lambda_id=5.0):
        self.device = device
        
        # Init Networks
        self.G_AB = ResNetGenerator(3, 3, n_blocks=6).to(device) # Sketch -> Photo
        self.G_BA = ResNetGenerator(3, 3, n_blocks=6).to(device) # Photo -> Sketch
        self.D_A = PatchGANDiscriminator(3).to(device)          # Sketch Discriminator
        self.D_B = PatchGANDiscriminator(3).to(device)          # Photo Discriminator
        
        init_weights(self.G_AB)
        init_weights(self.G_BA)
        init_weights(self.D_A)
        init_weights(self.D_B)
        
        # Loss Functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=lr, betas=(beta1, beta2)
        )
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=lr, betas=(beta1, beta2))
        
        # Mixed Precision
        self.scaler_G = GradScaler()
        self.scaler_D_A = GradScaler()
        self.scaler_D_B = GradScaler()
        
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id
        
        self.gen_losses = []
        self.disc_losses = []
        self.cycle_losses = []

    def train_step(self, real_A, real_B):
        # ------------------
        # Train Generators
        # ------------------
        self.optimizer_G.zero_grad()
        
        with autocast():
            # Identity loss (G_AB(B) should be B, G_BA(A) should be A)
            loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A) * self.lambda_id
            loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B) * self.lambda_id
            
            # GAN loss (G_AB(A) should look like B)
            fake_B = self.G_AB(real_A)
            loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B)))
            
            # GAN loss (G_BA(B) should look like A)
            fake_A = self.G_BA(real_B)
            loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))
            
            # Cycle loss
            rec_A = self.G_BA(fake_B)
            loss_cycle_A = self.criterion_cycle(rec_A, real_A) * self.lambda_cycle
            
            rec_B = self.G_AB(fake_A)
            loss_cycle_B = self.criterion_cycle(rec_B, real_B) * self.lambda_cycle
            
            # Total Generator Loss
            loss_G = loss_id_A + loss_id_B + loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B
            
        self.scaler_G.scale(loss_G).backward()
        self.scaler_G.step(self.optimizer_G)
        self.scaler_G.update()
        
        # --------------------
        # Train Discriminator A
        # --------------------
        self.optimizer_D_A.zero_grad()
        with autocast():
            # Real loss
            loss_real = self.criterion_GAN(self.D_A(real_A), torch.ones_like(self.D_A(real_A)))
            # Fake loss (using detached fake_A)
            loss_fake = self.criterion_GAN(self.D_A(fake_A.detach()), torch.zeros_like(self.D_A(fake_A.detach())))
            loss_D_A = (loss_real + loss_fake) * 0.5
        
        self.scaler_D_A.scale(loss_D_A).backward()
        self.scaler_D_A.step(self.optimizer_D_A)
        self.scaler_D_A.update()
        
        # --------------------
        # Train Discriminator B
        # --------------------
        self.optimizer_D_B.zero_grad()
        with autocast():
            # Real loss
            loss_real = self.criterion_GAN(self.D_B(real_B), torch.ones_like(self.D_B(real_B)))
            # Fake loss (using detached fake_B)
            loss_fake = self.criterion_GAN(self.D_B(fake_B.detach()), torch.zeros_like(self.D_B(fake_B.detach())))
            loss_D_B = (loss_real + loss_fake) * 0.5
            
        self.scaler_D_B.scale(loss_D_B).backward()
        self.scaler_D_B.step(self.optimizer_D_B)
        self.scaler_D_B.update()
        
        return {
            "loss_G": loss_G.item(),
            "loss_D": (loss_D_A + loss_D_B).item(),
            "loss_cycle": (loss_cycle_A + loss_cycle_B).item()
        }
