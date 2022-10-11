def __fix_channels(image):
    # if greyscale, repeat channels
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image


def __normalize_image(image):
    image = image / 255.0
    return image
