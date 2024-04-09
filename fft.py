import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
 
img = cv.imread('ants.jpg', cv.IMREAD_GRAYSCALE)

def fft(img):
    img = np.fft.fft2(img)
    img = np.fft.fftshift(img)
    return img

fimg = fft(img)
mag = 20 * np.log(np.abs(fimg))
plt.imshow(mag)
plt.show()