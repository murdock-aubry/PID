from PIL import Image

def crop_and_resize(image_path, output_size=(512, 512)):
    img = Image.open(image_path)

    width, height = img.size

    short_side = min(width, height)

    left = (width - short_side) // 2
    top = (height - short_side) // 2
    right = (width + short_side) // 2
    bottom = (height + short_side) // 2

    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize(output_size)

    return img_resized

image_path = 'original_image/leighia_test.jpg'
output_image = crop_and_resize(image_path)
output_image.save('prepared_image/leighia_test.jpg')  # To save the image