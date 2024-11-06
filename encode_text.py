import torch

def encode_text_to_image(text, image_size=11):
    # check if the text is too long
    max_text_length = image_size * image_size
    if len(text) > max_text_length:
        raise ValueError(f'Text is too long. Max text length is {max_text_length}')
    

    # Convert text to binary
    binary_text = ''.join(format(ord(char), '08b') for char in text)

    # Create an image tensor
    image = torch.zeros(image_size, image_size)

    # Fill the image tensor with the binary text
    for i, bit in enumerate(binary_text):
        row = i // image_size
        col = i % image_size
        image[row, col] = float(bit)

    return image


# Test the function
text = 'Hamish Burke'
image = encode_text_to_image(text)

# display the image
import matplotlib.pyplot as plt
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.savefig('text_image.png')

def decode_image_to_text(image):
    # Convert the image tensor to a binary string
    binary_text = ''.join(str(int(bit)) for row in image for bit in row)

    # Convert the binary string to text
    text = ''
    for i in range(0, len(binary_text), 8):
        byte = binary_text[i:i+8]
        text += chr(int(byte, 2))

    return text

# Test the function
decoded_text = decode_image_to_text(image)

print(f'Decoded text: {decoded_text}')
