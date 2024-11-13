import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

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

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = os.listdir(image_dir)
        self.transform = transform or transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

def train(model, dataloader, timesteps, betas, optimizer, criterion, device):
    model.train()
    for epoch in range(10):  # Set number of epochs
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

def sample(model, timesteps, betas, device):
    model.eval()
    with torch.no_grad():
        # Initialize with random noise
        x = torch.randn(1, 1, 256, 256, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
        for t in reversed(range(timesteps)):
            t_tensor = torch.tensor([t], device=device)
            eps_theta = model(x, t_tensor)
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = sqrt_recip_alphas[t] * (x - (betas[t] / sqrt_one_minus_alphas_cumprod[t]) * eps_theta) + noise * torch.sqrt(betas[t])
        return x

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
    train(model, dataloader, timesteps, betas, optimizer, criterion, device)
    
    # Generate a new image with the trained model
    generated_image = sample(model, timesteps, betas, device)
    # Convert tensor to image and save
    transforms.ToPILImage()(generated_image.squeeze().cpu()).save('generated_image_diffusion.png')
    print("Generated image saved as generated_image.png")

    print("Success")

if __name__ == "__main__":
    main()