import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import logging
import sys

# Add parent dir to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet_model import FusionUNet

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class PatchDataset(Dataset):
    def __init__(self, patches_dir, split='train'):
        self.patches_dir = os.path.join(patches_dir, split)
        self.files = [f for f in os.listdir(self.patches_dir) if f.endswith('_prod.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        image_path = os.path.join(self.patches_dir, file_name)
        label_path = image_path.replace('_prod.npy', '_label.npy')
        
        image = np.load(image_path)
        label = np.load(label_path)
        
        # Add channel dim to label if needed (for BCEWithLogits, usually (N,1,D,H,W))
        label = np.expand_dims(label, axis=0) 
        
        return torch.from_numpy(image), torch.from_numpy(label)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs: Logits (no sigmoid applied yet)
        # targets: Binary mask (0 or 1)
        
        # BCEWithLogitsLoss handles sigmoid internally for stability with AMP
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # For Dice, we need to apply sigmoid manually to get probabilities
        inputs_prob = torch.sigmoid(inputs)       
        
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)  
        
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class SegmentationTrainingAgent:
    def __init__(self, config_path="configs/config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.patches_dir = self.config["patches_dir"]
        self.models_dir = self.config["models_dir"]
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device}")
        
    def train(self, epochs=10, learning_rate=1e-4):
        # Data
        train_ds = PatchDataset(self.patches_dir, 'train')
        val_ds = PatchDataset(self.patches_dir, 'val')
        
        if len(train_ds) == 0:
            logging.error("No training data found in patches/train")
            return

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0) # Batch size 1 as requested
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
        
        # Model
        model = FusionUNet(in_channels=2, out_channels=1, init_features=16).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = DiceBCELoss() # Dice + BCE
        scaler = GradScaler() # Mixed Precision
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, labels in loop:
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.float32)
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                
                # VRAM Safety
                del images, labels, outputs, loss
                # Note: Regularly clearing cache can be slow, but requested by user
                # torch.cuda.empty_cache() 
            
            # Validation
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")
            logging.info(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(self.models_dir, "unet_best.pth"))
                print("Saved Best Model")
            
            torch.cuda.empty_cache()

if __name__ == "__main__":
    import torch.nn.functional as F # Re-import for Loss class if needed locally
    agent = SegmentationTrainingAgent()
    agent.train(epochs=5) # 5 epochs for demo/thesis start
