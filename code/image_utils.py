
import numpy as np
import matplotlib.pyplot as plt

def encode_text_to_image(text, image_size=256):
    # Convert the text to binary
    binary_text = ''.join(format(ord(char), '08b') for char in text)
    
    # Create an image
    image = np.zeros((image_size, image_size))
    
    # Fill the image with the text
    for i, bit in enumerate(binary_text):
        row = i // image_size
        col = i % image_size
        image[row, col] = float(bit)
    
    return image

def decode_image_to_text(image):
    # Convert the image tensor to a binary string
    binary_text = ''.join(str(int(bit)) for row in image for bit in row)
    
    # Convert the binary string to text
    text = ''
    for i in range(0, len(binary_text), 8):
        byte = binary_text[i:i+8]
        text += chr(int(byte, 2))
    
    return text


def load_images(path):
    from PIL import Image
    import os
    
    images = []
    image_paths = os.listdir(path)
    for image_path in image_paths:
        image = Image.open(os.path.join(path, image_path))
        images.append(image)
    return images

def make_image_grid(images, nrow=8):
    # Get the number of images
    n_images = len(images)
    
    # Calculate the number of rows
    ncol = (n_images - 1) // nrow + 1
    
    # create it manually
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    for i, image in enumerate(images):
        row = i // ncol
        col = i % ncol
        ax[row, col].imshow(image, cmap='gray')
        ax[row, col].axis('off')
    for i in range(n_images, nrow * ncol):
        row = i // ncol
        col = i % ncol
        ax[row, col].axis('off')
    plt.tight_layout()
    return fig

def show_image(image):
    import torch
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise ValueError('Image must be a PyTorch tensor or NumPy array')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def save_image(image, filename):
    plt.imsave(filename, image, cmap='gray')