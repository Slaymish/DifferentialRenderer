import torch
import torch.nn as nn

class CNNFactory(nn.Module):
    def __init__(self, activation=nn.ReLU(), seed=69, end_activation=nn.Softmax(dim=1)):
        super(CNNFactory, self).__init__()
        torch.manual_seed(seed)
        self.seed = seed
        self.activation = activation
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            self.activation,
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 10),  # Adjust if input size changes
            end_activation
        )

    def set_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)

    def get_live_model(self):
        return self.model
    
    def set_input_channels(self, channels):
        self.model[0] = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1)
        self.propogate_channels(self.model, channels)

    def propogate_channels(self, model, channels):
        for layer in model:
            if hasattr(layer, "in_channels"):
                layer.in_channels = channels

def test_factory(factory):
    print("=== Testing factory with seed:", factory.seed, "and activation:", factory.activation.__class__.__name__, "===")
    model = factory.get_live_model()

    # get size of input tensor
    input_size = model[0].in_channels
    print("Input size: ", input_size)

    model.eval()

    # Test forward pass

    x = torch.randn(1, input_size, 32, 32)
    y = model(x)
    loss_fn = nn.CrossEntropyLoss()
    target = torch.empty(1, dtype=torch.long).random_(10)
    loss = loss_fn(y, target)
    print("Loss: ", loss.item())

    print("Model Structure:")
    print(model)
    print("=== End of test ===\n")


def quick_train_eval(model, epochs=10):
    print("=== Starting quick train eval ===")
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []

    input_size = model[0].in_channels  # Get input channels from the model

    for epoch in range(epochs):
        optimizer.zero_grad()
        x = torch.randn(1, input_size, 32, 32)  # Use dynamic input channels
        y = model(x)
        target = torch.empty(1, dtype=torch.long).random_(10)
        loss = loss_fn(y, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    x = torch.randn(1, input_size, 32, 32)  # Use dynamic input channels
    y = model(x)
    print("Output: ", y)
    print("Loss: ", losses[-1])

    print("=== End of quick train eval ===\n")

def grid_test():
    print("=== Starting Grid Test ===")
    for i in range(10):
        for activation in [nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid()]:
            model = CNNFactory(activation=activation, seed=i)
            test_factory(model)
    print("=== End of Grid Test ===")

def main():
    model1 = CNNFactory(activation=nn.ReLU(), seed=420)

    model1.set_input_channels(1)

    model2 = CNNFactory(activation=nn.LeakyReLU(),end_activation=nn.Sigmoid(), seed=69)

    test_factory(model1)

    quick_train_eval(model1.get_live_model())


if __name__ == "__main__":
    main()
