from skimage.color import rgb2yuv
from cv2 import resize, INTER_LINEAR
from matplotlib import pyplot as plt
import scipy.misc
def scale_image(x):
    # Extract the Y component after converting to YUV
    x_yuv = rgb2yuv(x)
    x_y = x_yuv[:, :, 0]

    # Scale to input size for convnet
    temp = resize(x_y, (84, 84), interpolation=INTER_LINEAR)
    scipy.misc.toimage(temp).save('outfile.jpg')

    return resize(x_y, (84, 84), interpolation=INTER_LINEAR)
