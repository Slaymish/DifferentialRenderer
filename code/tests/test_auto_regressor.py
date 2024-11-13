import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from models.auto_regressor import AutoRegressor


def generate_image(model, device, img_size=(32, 32)):
    model.eval()
    with torch.no_grad():
        image = torch.zeros(1, 3, *img_size).to(device)
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                output = model(image)
                # Extract the pixel value at (i, j)
                pixel_value = torch.clamp(output[:, :, i, j], 0.0, 1.0)
                image[:, :, i, j] = pixel_value
        return image.cpu().squeeze(0)
    
def make_bottom_half_of_image_black(image):
    # Clone the image to avoid in-place modification
    image_copy = image.clone()
    image_copy[:, 16:, :] = 0
    return image_copy
    
def test_saved_model():
    model = AutoRegressor()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)

    # Generate multiple images for the grid
    images = []
    for _ in range(12):
        image = gen_image(model, device)
        images.append(image)

    # Stack images to create a grid
    images = torch.stack(images)  # Shape: (12, 3, 32, 32)

    # Reshape to form a 3x4 grid (3 rows, 4 columns)
    images = images.view(3, 4, 3, 32, 32)  # Shape: (3, 4, 3, 32, 32)
    images = images.permute(0, 3, 1, 4, 2)  # Shape: (3, 32, 4, 32, 3)
    images = images.reshape(96, 128, 3)  # Final grid shape: (96, 128, 3)

    # Convert to numpy array and display the image grid
    images = images.numpy()

    # Plot and save the grid of generated images
    plt.imshow(images)
    plt.axis("off")
    plt.show()

    plt.imsave("generated_images.png", images)

def gen_image(model, device):
    model.eval()
    with torch.no_grad():
        image = torch.zeros(1, 3, 32, 32).to(device)
        for i in range(32):
            for j in range(32):
                output = model(image)
                pixel_value = torch.clamp(output[:, :, i, j], 0.0, 1.0)
                image[:, :, i, j] = pixel_value
        return image.cpu().squeeze(0)




def complete_image(model, device, image):
    model.eval()
    with torch.no_grad():
        # Clone the image to avoid modifying the original
        diff_image = image.clone().unsqueeze(0).to(device)  # Add batch dimension
        for i in range(16, 32):
            for j in range(32):
                output = model(diff_image)
                pixel_value = torch.clamp(output[:, :, i, j], 0.0, 1.0)
                diff_image[:, :, i, j] = pixel_value
    return diff_image.cpu().squeeze(0)  # Remove batch dimension


if __name__ == "__main__":
    images = load_images("images")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    # Get three random images and cut them in half
    images = [transform(image) for image in images]
    import random
    three_random_ids = random.sample(range(len(images)), 3)
    images = [images[i] for i in three_random_ids]
    images_blackened = [make_bottom_half_of_image_black(image) for image in images]

    model = AutoRegressor()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)

    completed_images = [complete_image(model, device, image) for image in images_blackened]

    # Display the original and completed images
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i, (original, completed) in enumerate(zip(images_blackened, completed_images)):
        axes[0, i].imshow(original.permute(1, 2, 0))
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Original Image {i+1}")

        axes[1, i].imshow(completed.permute(1, 2, 0))
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Completed Image {i+1}")

    plt.tight_layout()
    plt.show()

    plt.savefig("completed_images.png")