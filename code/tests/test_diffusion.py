import torch
from torchvision import transforms

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

if __name__ == '__main__':
    # Generate a new image with the trained model
    generated_image = sample(model, timesteps, betas, device)
    # Convert tensor to image and save
    transforms.ToPILImage()(generated_image.squeeze().cpu()).save('generated_image_diffusion.png')
    print("Generated image saved as generated_image.png")