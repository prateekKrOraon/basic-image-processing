import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

# Sobel edge operators

# horizontal
sobel_h = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
]

# vertical
sobel_v = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]


def pad(img):
    shape = img.shape
    rows = 0
    cols = 0
    channels = 0
    if len(shape)>2:
        rows, cols, channels = shape
    else:
        rows, cols = shape

    padded = np.zeros((rows + 2, cols + 2, channels))
    for i in range(rows + 2):
        for j in range(cols + 2):
            if channels != 0:
                for c in range(channels):
                    if i != 0 and i != rows + 1 and j != 0 and j != cols + 1:
                        padded[i, j, c] = img[i - 1, j - 1, c]
            else:
                if i != 0 and i != rows + 1 and j != 0 and j != cols + 1:
                    padded[i, j, c] = img[i - 1, j - 1, c]

    plt.imshow(padded)
    plt.title("Padded Image")
    plt.show()

    return padded


def apply_sobel(img, kernel):
    padded = pad(img)
    rows, cols, channels = padded.shape
    new_img = np.zeros((rows - 2, cols - 2, channels))
    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                pixel = 0
                if i < rows - 2 and j < cols - 2:
                    for k in range(3):
                        for l in range(3):
                            pixel = pixel + kernel[k][l] * padded[i + k, j + l, c]
                    new_img[i, j, c] = round(pixel)

    plt.imshow(new_img)
    plt.title("Applying Sobel Kernel")
    plt.show()

    return new_img


def mag_and_dir(img_h, img_v):
    rows, cols, channels = img_h.shape
    magnitude = np.zeros((rows, cols, channels))
    direction = np.zeros((rows, cols, channels))

    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                pixel = math.sqrt((img_h[i, j, c] ** 2) + (img_v[i, j, c] ** 2))
                if pixel <= 0:
                    magnitude[i, j, c] = 0
                elif pixel > 255:
                    magnitude[i, j, c] = 255
                elif pixel is math.nan:
                    magnitude[i, j, c] = 0
                else:
                    magnitude[i, j, c] = round(pixel)

                pixel = math.atan(img_v[i, j, c] / img_h[i, j, c]) * 255
                if pixel <= 0:
                    direction[i, j, c] = 0
                elif pixel > 255:
                    direction[i, j, c] = 255
                elif pixel is np.NaN:
                    direciton[i, j, c] = 0
                else:
                    direction[i, j, c] = pixel

    plt.imshow(magnitude)
    plt.title("Magnitude")
    plt.show()
    plt.imshow(direction)
    plt.title("Direction")
    plt.show()

    return [magnitude, direction]


def calculate_mean(group):
    total = 0
    rows, cols, channels = group.shape
    count = rows * cols * channels
    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                total = total + group[i, j, c]

    mean = total / count
    return mean


def calc_threshold(img):
    t = 0
    new_t = 150
    rows, cols, channels = img.shape
    g_1 = np.zeros((rows, cols, channels))
    g_2 = np.zeros((rows, cols, channels))
    count = 0
    while (abs(new_t - t) > 5):
        t = new_t
        for i in range(rows):
            for j in range(cols):
                for c in range(channels):
                    pixel = img[i, j, c]
                    if pixel > t:
                        g_2[i, j, c] = pixel
                    else:
                        g_1[i, j, c] = pixel

        mean_1 = calculate_mean(g_1)
        print("mean 1 = {}".format(mean_1))
        mean_2 = calculate_mean(g_2)
        print("mean 2 = {}".format(mean_2))
        new_t = (mean_1 + mean_2) / 2
        print("threshold = {}".format(new_t))
        count += 1
        print("Iterations = {}".format(count))
        print("Difference = {}".format(abs(new_t - t)))

    return [new_t, g_1, g_2]


def main():
    print("importin image: images/image1.jpg")
    img = cv.imread('images/image1.jpg')

    print("horizontal_sobel")
    sobel_img_h = apply_sobel(img, sobel_h)
    cv.imwrite('output/horizontal_sobel.jpg', sobel_img_h)

    print("vertical_sobel")
    sobel_img_v = apply_sobel(img, sobel_v)
    cv.imwrite('output/verticle_sobel.jpg', sobel_img_v)

    print("magnitude and direction")
    magnitude, direction = mag_and_dir(sobel_img_h, sobel_img_v)
    cv.imwrite('output/magnitude.jpg', magnitude)
    cv.imwrite('output/direction.jpg', direction)

    print("Thresholding")
    mag = cv.imread('output/magnitude.jpg')
    a = calc_threshold(mag)

    histr = cv.calcHist([mag], [0], None, [256], [0, 256])
    plt.plot(histr)
    plt.plot("histogram")
    plt.show()

    plt.imshow(a[1])
    plt.title("Group 1 intensity <= {}".format(a[0]))
    plt.show()

    plt.imshow(a[2])
    plt.title("Group 2 intensity > {}".format(a[0]))
    plt.show()

    cv.imwrite('output/global_thresholding/group_1.jpg', a[1])
    cv.imwrite('output/global_thresholding/group_2.jpg', a[2])


if __name__ == "__main__":
    main()