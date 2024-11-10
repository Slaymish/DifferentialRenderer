import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from image_utils import load_images
import os

# [Include MaskedConv2d and AutoRegressor classes as defined above]
class MaskedConv2d(torch.nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in ['A', 'B'], "mask_type must be 'A' or 'B'"
        self.mask_type = mask_type
        self.register_buffer('mask', self.weight.data.clone())
        self.create_mask()

    def create_mask(self):
        kH, kW = self.kernel_size
        self.mask.fill_(1)
        yc, xc = kH // 2, kW // 2

        # For Mask A, zero out the center pixel
        if self.mask_type == 'A':
            self.mask[:, :, yc, xc:] = 0
            self.mask[:, :, yc+1:, :] = 0
        else:
            self.mask[:, :, yc, xc+1:] = 0
            self.mask[:, :, yc+1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class AutoRegressor(torch.nn.Module):
    def __init__(self):
        super(AutoRegressor, self).__init__()
        self.conv1 = MaskedConv2d('A', 3, 64, kernel_size=7, padding=3)
        self.conv2 = MaskedConv2d('B', 64, 64, kernel_size=7, padding=3)
        self.conv3 = MaskedConv2d('B', 64, 64, kernel_size=7, padding=3)
        self.conv4 = MaskedConv2d('B', 64, 256, kernel_size=7, padding=3)
        self.conv_out = torch.nn.Conv2d(256, 3, kernel_size=1)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.conv_out(x)
        return x


def prepare_data(images):
    inputs = []
    targets = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for image in images:
        image = image.to(device)
        # No need to shift images since masking handles dependencies
        inputs.append(image)
        targets.append(image)
    return inputs, targets

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

def main():
    model = AutoRegressor()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load and preprocess images
    images = load_images("images")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    if not images:
        print("No images found")
        return
    else:
        print(f"Found {len(images)} images")

    images = [transform(image) for image in images]
    images = images[:2]  # Limit to 2 images for demonstration

    inputs, targets = prepare_data(images)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    losses = []
    epochs = 10000

    for epoch in range(epochs):
        epoch_loss = 0
        for input_image, target_image in zip(inputs, targets):
            input_image = input_image.unsqueeze(0).to(device)
            target_image = target_image.unsqueeze(0).to(device)

            # Forward pass
            output = model(input_image)

            # Compute loss
            loss = criterion(output, target_image)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(inputs)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    # Save the model
    torch.save(model.state_dict(), "model.pth")

    # Plot losses
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.savefig("loss.png")
    plt.show()

    # Generate an image
    generated_image = generate_image(model, device)
    # Convert tensor to PIL Image
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(generated_image)
    pil_image.save("generated_image.png")
    pil_image.show()
    

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


def make_bottom_half_of_image_black(image):
    # Clone the image to avoid in-place modification
    image_copy = image.clone()
    image_copy[:, 16:, :] = 0
    return image_copy

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
