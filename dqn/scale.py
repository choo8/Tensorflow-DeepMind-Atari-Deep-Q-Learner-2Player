from skimage.color import rgb2yuv
from cv2 import resize, INTER_LINEAR


def scale_image(x):
    # Extract the Y component after converting to YUV
    x_yuv = rgb2yuv(x)
    x_y = x_yuv[:, :, 0]

    # Scale to input size for convnet
    return resize(x_y, (84, 84), interpolation=INTER_LINEAR)
