import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from utils.image_utils import load_images
import utils.constants as constants

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
    torch.save(model.state_dict(), constants.MODEL_SAVE_DIR + "/auto_regressor.pth")

    # Plot losses
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.savefig("loss.png")
    plt.show()

    
    
if __name__ == "__main__":
    main()