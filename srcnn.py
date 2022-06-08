import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
# import cv2


def imread(path, is_grayscale=True):
    """
    Read image from the giving path.
    Default value is gray-scale, and image is read by YCbCr format as the paper.
    """
    if is_grayscale:
        return iio.imread(path, as_gray=True, pilmode='YCbCr').astype(np.float32)
    else:
        return iio.imread(path, pilmode='YCbCr').astype(np.float32)


def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)  # remainder of height
        w = w - np.mod(w, scale)  # remainder of width
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def preprocess(path, scale=3):
    """
    Preprocess single image file
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with interpolation
    Args:
      path: file path of desired file
      input_: image applied interpolation (low-resolution)
      label_: image with original resolution (high-resolution), groundtruth
    """
    image = imread(path, is_grayscale=True)
    label_ = modcrop(image, scale)

    # Must be normalized As the pixel values range from 0 to 256, apart from 0 the range is 255.
    # So dividing all the values by 255 will convert it to range from 0 to 1.
    label_ = label_ / 255.

    # Zooming refers to increase the quantity of pixels, so that when you zoom an image,
    # you will see more detail. Interpolation works by using known data to estimate values at unknown points.
    input_ = ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
    input_ = ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)

    return input_, label_




## ------ Add your code here: set the weight of three conv layers
# replace 'None' with your hyper parameter numbers
# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=6, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=0, bias=True)
        # print("The weight of the 1st filter is:"np.squeeze(self.conv1.weight[0]))
        # print("The bias of the 9th filter is:",np.squeeze(self.conv1.bias[9]))


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return out

"""Load the pre-trained model file
"""
model = SRCNN()
model.load_state_dict(torch.load('./model/model.pth'))
model.eval()

# """Show the weights of the convolutions"""
# weight = model.conv1.weight.detach().numpy()
# plt.imshow(weight[0])

"""Read the test image
"""
lr_image, hr_image = preprocess('./image/butterfly_GT.bmp')

# imgplot = plt.imshow(lr_image)
# plt.show()
#
# ground_truth=iio.imread('./image/butterfly_GT.bmp', as_gray=True, pilmode='YCbCr').astype(np.float32)
# """Show output image"""

# imgplot = plt.imshow(ground_truth)
# plt.show()

#Read an image
# image = iio.imread('./image/butterfly_GT.bmp', as_gray=True, pilmode='YCbCr').astype(np.float32)

#display the Resolution of the image
# wid = image.shape[1]
# hgt = image.shape[0]
#
# # displaying the dimensions
# print('The resolutions of the image are:', str(wid) + "x" + str(hgt))

# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(lr_image, axis=0), axis=0)
input_ = torch.from_numpy(input_)

"""Run the model and get the SR image
"""
with torch.no_grad():
    output_ = model(input_)

output_ = output_.numpy().squeeze((0, 1))

# """Show output image"""
#
# imgplot = plt.imshow(output_)
# plt.show()

##------ Add your code here: save the LR and SR images and compute the psnr
# # hints: use the 'iio.imsave()'  and ' sk image.metrics.peak_signal_noise_ratio()'


"""Save the LR and SR images 
"""

iio.imsave('./image/SR_image.bmp', output_)
iio.imsave('./image/LR_image.bmp', lr_image)
iio.imsave('./image/HR_image.bmp', hr_image)



"""Compute the PSNR metrics
"""
import skimage
psnr1 = skimage.metrics.peak_signal_noise_ratio(hr_image, output_)
print(f'PSNR between super resolution output and hr image: {psnr1}')

psnr2 = skimage.metrics.peak_signal_noise_ratio(hr_image, lr_image)
print(f'PSNR between lr and hr image: {psnr2}')


# psnr3= skimage.metrics.peak_signal_noise_ratio(hr_image, hr_image)
# print(f'PSNR between hr image and ground truth: {psnr3}')

# """Compare PSNR of baseline interpolation and SRCNN method"""
# perf= (psnr1-psnr2)/psnr1
# print(f"SCRNN method yields a performance increased by {perf*100}%")
