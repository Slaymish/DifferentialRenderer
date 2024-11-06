from encode_text import encode_text_to_image, decode_image_to_text

texts = ['Hello, World!', 'This is a test.', 'Another test.']

issues = 0

for text in texts:
    # Encode text to image
    try:
        image = encode_text_to_image(text)
    except ValueError as e:
        print(f'Error: {e}')
        issues += 1
        continue

    # Decode image to text
    decoded_text = decode_image_to_text(image)

    if text != decoded_text:
        print(f'Error: Decoded text does not match original text')
        print(f'Original text: {text}')
        print(f'Decoded text: {decoded_text}')
        issues += 1

if issues == 0:
    print('All tests passed successfully')
else:
    print(f'{issues} issues found')