import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def fft(img):
    img = np.fft.fft2(img)
    img = np.fft.fftshift(img)
    return img

def ifft(fimg):
    fimg = np.fft.ifftshift(fimg)
    fimg = np.fft.ifft2(fimg)
    return fimg

def mag(img):
    absvalue = np.abs(img)
    magnitude = 20 * np.log(absvalue)
    return magnitude

def norm(img):
    img = cv.normalize(
        img, None, 0, 1.0,
        cv.NORM_MINMAX
    )
    return img

def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def open(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return img

def bin(img):
    threshold, img = cv.threshold(
        img, 110, 255, cv.THRESH_OTSU
    )
    return img

def dilate(img, x, y):
    img = cv.dilate(img, np.ones((x, y)))
    return img

def erode(img, x, y):
    img = cv.erode(img, np.ones((x, y)))

def p1(img):
    img = bin(img)
    img = dilate(img, 4, 4)
    img = fft(img)
    img = mag(img)
    img = norm(img)
    return img
    
def p2(img):
    img = fft(img)
    img = mag(img)
    img = norm(img)
    return img
    
def p3(img):
    img = bin(img)
    img = dilate(img, 4, 4)
    return img

model = models.Sequential([
    layers.Resizing(64, 64),
    layers.Rescaling(1.0/255),
    layers.Conv2D(32, (5, 5),
        activation = 'relu',
        kernel_initializer = initializers.RandomNormal()
    ),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (5, 5),
        activation = 'relu',
        kernel_initializer = initializers.RandomNormal()
    ),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128,
        activation = 'relu',
        kernel_initializer = initializers.RandomNormal()
    ),
    layers.Dense(96,
        activation = 'relu',
        kernel_initializer = initializers.RandomNormal()
    ),
    layers.Dense(96,
        activation = 'relu',
        kernel_initializer = initializers.RandomNormal()
    ),
    layers.Dense(128,
        activation = 'relu',
        kernel_initializer = initializers.RandomNormal()
    ),
    layers.Dense(6,
        activation = 'sigmoid',
        kernel_initializer = initializers.RandomNormal()
    )
])