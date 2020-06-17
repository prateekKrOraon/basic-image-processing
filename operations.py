import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

max_intensity = 255


def convert_to_greyscale(img):
    b, g, r = cv.split(img)
    rows, cols = b.shape
    img_out = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            intensity = int((int(b[i][j]) + int(g[i][j]) + int(g[i][j])) / 3)
            img_out[i, j] = intensity

    return img_out


def plot_histogram(img):
    img = convert_to_greyscale(img)
    hist = [0] * 256
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            intensity = int(img[i][j])
            hist[intensity] = hist[intensity] + 1

    x = range(len(hist))
    plt.bar(x, hist)
    plt.show()


def equalize(img):
    hist = [0] * 256
    rows, cols, channel = img.shape
    for i in range(rows):
        for j in range(cols):
            intensity = int(round((img[i][j][0] + img[i][j][1] + img[i][j][2]) / 3))
            hist[intensity] = hist[intensity] + 1

    cumulative = [0] * len(hist)
    for i in range(len(hist)):
        if i == 0:
            cumulative[i] = hist[i]
        else:
            cumulative[i] = cumulative[i - 1] + hist[i]

    total = cumulative[len(cumulative) - 1]
    new_hist = [0] * len(cumulative)
    for i in range(len(cumulative)):
        grey_level = round((cumulative[i] / total) * max_intensity)
        new_hist[grey_level] = new_hist[grey_level] + hist[i]

    x = range(len(hist))
    plt.plot(x, new_hist)
    plt.show()


def negative_image(img):
    rows, cols, channel = img.shape
    new_img = np.zeros((rows,cols,channel))
    for i in range(rows):
        for j in range(cols):
            new_img[i,j,0] = max_intensity - img[i,j,0]
            new_img[i,j,1] = max_intensity - img[i,j,1]
            new_img[i,j,0] = max_intensity - img[i,j,2]
            if new_img[i,j,0] < 0:
                new_img[i,j,0] = 0
            if new_img[i,j,1] < 0:
                new_img[i,j,1] = 0
            if new_img[i,j,2] < 0:
                new_img[i,j,2] = 0

    return new_img


def log_transform(img, c):
    img = convert_to_greyscale(img)
    rows, cols = img.shape
    new_img = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            intensity = round(c * math.log(1 + img[i, j]))
            if intensity > 255:
                intensity = 255
            new_img[i, j] = intensity

    return new_img


def power_transform(img, c, g):
    img = convert_to_greyscale(img)
    rows, cols = img.shape
    new_img = np.zeros((rows, cols))
    minn = 99999999999999
    maxx = 0

    for i in range(rows):
        for j in range(cols):
            s = (c * img[i][j]) ** g
            if (s > maxx):
                maxx = s
            if (s < minn):
                minn = s

    for i in range(rows):
        for j in range(cols):
            s = (c * img[i][j]) ** g
            s = (s - minn) / (maxx - minn)
            s = s * 255
            new_img[i, j] = s

    return new_img


def contrast_stretch(img):
    rows, cols, channels = img.shape
    m_min = 256
    m_max = -1

    for i in range(rows):
        for j in range(cols):
            intensity = (img[i][j][0] + img[i][j][1] + img[i][j][2]) / 3
            if intensity > m_max:
                m_max = intensity
            if intensity < m_min:
                m_min = intensity

    new_img = np.zeros((rows, cols, channels))
    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                intensity = round(((img[i, j, c] - m_min) / (m_max - m_min)) * 255)
                if intensity < 0:
                    intensity = 0
                new_img[i, j, c] = intensity

    return new_img


# assuming 8 bit image
def bit_plane_slice(img, bit_plane):
    rows, cols, channels = img.shape
    img_size = (rows, cols)
    bit_size = img_size + (8,)

    b, g, r = cv.split(img)

    b_bits = np.unpackbits(b).reshape(bit_size)
    g_bits = np.unpackbits(g).reshape(bit_size)
    r_bits = np.unpackbits(r).reshape(bit_size)

    if bit_plane is not None:
        b_bits[:, :, int(bit_plane)] = 0
        g_bits[:, :, int(bit_plane)] = 0
        r_bits[:, :, int(bit_plane)] = 0

    b_aug = np.packbits(b_bits).reshape(img_size)
    g_aug = np.packbits(g_bits).reshape(img_size)
    r_aug = np.packbits(r_bits).reshape(img_size)

    return cv.merge((b_aug, g_aug, r_aug))