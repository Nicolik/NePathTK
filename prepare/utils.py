import cv2
from PIL import Image


def pad_if_needed(image_cv, target_size=1024):
    """
    Pads the input image on the right and bottom if its dimensions are less than target_size.
    Keeps the original content top-left aligned.

    Parameters:
        image_cv (np.ndarray): Input image (as loaded with cv2).
        target_size (int): Desired width and height (default: 1024).

    Returns:
        np.ndarray: Padded image with shape (target_size, target_size, channels)
    """
    h, w = image_cv.shape[:2]

    if h < target_size or w < target_size:
        pad_bottom = max(0, target_size - h)
        pad_right = max(0, target_size - w)

        image_cv = cv2.copyMakeBorder(
            image_cv, 0, pad_bottom, 0, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black padding
        )

    return image_cv


def pad_to_1024(img):
    width, height = img.size
    if width == 1024 and height == 1024:
        return None  # Signal that no padding is needed
    new_img = Image.new(img.mode, (1024, 1024))
    new_img.paste(img, (0, 0))
    return new_img
