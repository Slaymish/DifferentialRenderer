import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from utils.image_utils import ImageDataset
import utils.constants as constants

class DiffusionModel(torch.nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        # Updated to U-Net architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x, t):
        # t can be used to condition the model
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def train(model, dataloader, timesteps, betas, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        for x in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, timesteps, (1,), device=device).item()
            noise = torch.randn_like(x)
            x_noisy = x + betas[t] * noise
            y = model(x_noisy, t)
            loss = criterion(y, x)
            loss.backward()
            optimizer.step()
            print("Loss: ", loss.item())

    # Save the model
    torch.save(model.state_dict(), os.path.join(constants.MODEL_SAVE_DIR, "diffusion_model.pth"))



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionModel().to(device)

    transformers = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    # Define dataset and dataloader
    dataset = ImageDataset('images/', transform=transformers)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Define noise schedule
    timesteps = 1000
    betas = torch.linspace(0.0001, 0.02, timesteps).to(device)
    
    # Train the model
    train(model, dataloader, timesteps, betas, optimizer, criterion, device, epochs=10)
    
    

if __name__ == "__main__":
    main()