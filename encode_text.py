import torch
import qrcode
from image_utils import QR_implementation, make_image_grid, show_image, save_image

# literaly just qr code basically
# Encode a text message into a binary image
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


if __name__ == "__main__":
    # Encode a text message
    data = ['Hello, world!', 'This is a test.', 'This is a longer test.', 'This ioujdvfojodvjodajojaojfojaofjojdojfajoifjadjofjadiojfoiajojdaojfoajis a very long test.']
    import qrcode as qr
    patterns = [qr.ERROR_CORRECT_L, qr.ERROR_CORRECT_L, qr.ERROR_CORRECT_L, qr.ERROR_CORRECT_L]
    images = []
    for text, pattern in zip(data, patterns):
        image = QR_implementation(text, version=1, error_correction=pattern, mask_pattern=0)
        images.append(image)

    # Display the images
    imagegrid = make_image_grid(images, nrow=2)
    print(type(imagegrid))
    imagegrid.show()
    imagegrid.savefig('qr_codes.png')
    #show_image(imagegrid)
    #save_image(imagegrid, 'qr_codes.png')


