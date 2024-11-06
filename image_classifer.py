import torch
import matplotlib.pyplot as plt

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

# Create a model
conv_layers = [(6, 5), (16, 5)]  # (out_channels, kernel_size)
fc_layers = [120, 84]
model = ImageClasifier(2, conv_layers, fc_layers)  # binary classification

# create param dict (going to grid search), and try each model  
activation_fns = [torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.Softmax]
dropout_rates = [0.1, 0.3, 0.5, 0.7]
conv_layers = [[(6, 5), (16, 5)], [(6, 5), (16, 5), (32, 5)], [(6, 5), (16, 5), (32, 5), (64, 5)]]
fc_layers = [[120, 84]]

def unhash_params(hashed_params):
    return eval(hashed_params)

def hash_params(params):
    return str(params)

def grid_search(activation_fns, dropout_rates, conv_layers, fc_layers):
    results = {}
    for activation_fn in activation_fns:
        for dropout_rate in dropout_rates:
            for conv_layer in conv_layers:
                for fc_layer in fc_layers:
                    try:
                        model = ImageClasifier(2, conv_layer, fc_layer, dropout_rate, activation_fn)
                        print(f'Training model with activation_fn={activation_fn}, dropout_rate={dropout_rate}, conv_layer={conv_layer}, fc_layer={fc_layer}')
                        val_losses, test_losses = train_model(model)
                        results[hash_params((activation_fn, dropout_rate, conv_layer, fc_layer))] = (val_losses, test_losses)
                    except Exception as e:
                        print(f'Error training model with activation_fn={activation_fn}, dropout_rate={dropout_rate}, conv_layer={conv_layer}, fc_layer={fc_layer}')
                        print(e)
    return results

def train_model(model):
    # create synthetic data
    target = torch.tensor([1])
    x_val = torch.randn(1, 3, 32, 32)
    x_test = torch.randn(1, 3, 32, 32)

    # test backward pass
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    val_losses = []
    test_losses = []

    for i in range(100):
        optimizer.zero_grad()
        y_val = model(x_val)
        val_loss = criterion(y_val, target)
        val_loss.backward()
        optimizer.step()
        val_losses.append(val_loss.item())

        y_test = model(x_test)
        test_loss = criterion(y_test, target)
        test_losses.append(test_loss.item())

        if i % 100 == 0:
            print(f'Iteration {i}, Validation Loss: {val_loss.item()}, Test Loss: {test_loss.item()}')

    return val_losses, test_losses

texts = ['Hello, World!', 'This is a test.', 'Another test.', 'This is a very long text that will raise an error.']

from encode_text import encode_text_to_image, decode_image_to_text

images = []
image_dim = max(len(text) for text in texts)
for text in texts:
    try:
        image = encode_text_to_image(text, image_dim)
        images.append(image)
    except ValueError as e:
        print(f'Error: {e}')


# create model using image_dim
model = ImageClasifier(2, [(6, 5), (16, 5)], [120, 84])

# test forward pass
for image in images:
    image = image.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    output = model(image)
    print(output)

# test backward pass
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

target = torch.tensor([1])
for i in range(100):
    optimizer.zero_grad()
    for image in images:
        image = image.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
    optimizer.step()
    print(f'Iteration {i}, Loss: {loss.item()}')
