import torch
from torchvision import datasets, transforms
from image_utils import load_images
import numpy as np

datasets.DatasetFolder

class ImageClasifier(torch.nn.Module):
    def __init__(self, num_classes, conv_layers, fc_layers, dropout_rate=0.5, activation_fn=torch.nn.ReLU):
        super(ImageClasifier, self).__init__()
        self.num_classes = num_classes
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn

        layers = []
        in_channels = 3
        for out_channels, kernel_size in conv_layers:
            layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size))
            layers.append(torch.nn.MaxPool2d(2, 2))
            layers.append(activation_fn())
            in_channels = out_channels

        self.conv = torch.nn.Sequential(*layers)

        fc_input_dim = self._get_conv_output_dim()
        fc_layers_list = []
        for fc_dim in fc_layers:
            fc_layers_list.append(torch.nn.Linear(fc_input_dim, fc_dim))
            fc_layers_list.append(activation_fn())
            fc_layers_list.append(torch.nn.Dropout(dropout_rate))
            fc_input_dim = fc_dim

        fc_layers_list.append(torch.nn.Linear(fc_input_dim, num_classes))
        self.fc = torch.nn.Sequential(*fc_layers_list)

    def _get_conv_output_dim(self):
        x = torch.randn(1, 3, 32, 32)
        x = self.conv(x)
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)




def train_epoch(model, optimizer, criterion, x, target):
    for i in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def train_model(model, strawberries_loader, tomatoes_loader,epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        for i, data in enumerate(strawberries_loader):
            x, target = data
            train_epoch(model, optimizer, criterion, x, target)

        val_losses = []
        for i, data in enumerate(tomatoes_loader):
            x, target = data
            output = model(x)
            loss = criterion(output, target)
            val_losses.append(loss.item())

        test_losses = []
        for i, data in enumerate(strawberries_loader):
            x, target = data
            output = model(x)
            loss = criterion(output, target)
            test_losses.append(loss.item())

    return val_losses, test_losses


class ImageDataSet(datasets):
    def __init__(self,s_images,t_images,train_size=0.8):
        self.transformers = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        # lists of ImageFile
        self.images = np.stack(s_images,t_images)

    def set_transforms(self,transformers):
        self.transformers = transformers


    def get_idx(self,idx):
        return self.transformers(self.images[idx])




def main():
    # Create a model
    conv_layers = [(6, 5), (16, 5)]  # (out_channels, kernel_size)
    fc_layers = [120, 84]
    model = ImageClasifier(2, conv_layers, fc_layers)  # binary classification

    strawberries = "strawberry_images"
    tomatoes = "tomato_images"

    s_images = load_images(strawberries)
    t_images = load_images(tomatoes)

    train_size = 0.8

    # combine s and t


    strawberries_dataset = datasets.ImageFolder(strawberries, transform=transforms.ToTensor())
    tomatoes_dataset = datasets.ImageFolder(tomatoes, transform=transforms.ToTensor())

    strawberries_loader = torch.utils.data.DataLoader(strawberries_dataset, batch_size=4, shuffle=True)
    tomatoes_loader = torch.utils.data.DataLoader(tomatoes_dataset, batch_size=4, shuffle=True)

    val_losses, test_losses = train_model(model, strawberries_loader, tomatoes_loader)

    print(val_losses)

    if strawberries:
        print(strawberries)
    else:
        print("dont")
    print(test_losses)


if __name__ == "__main__":
    main()