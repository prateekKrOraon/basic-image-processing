import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math


def pad(img):
    rows, cols = img.shape
    padded_img = np.zeros((rows + 2, cols + 2))

    for i in range(rows + 2):
        for j in range(cols + 2):
            if i < rows and j < cols:
                padded_img[i + 1, j + 1] = img[i, j]

    cv.imwrite('output/padded_image2.jpg', padded_img)
    return padded_img


def return_i(num, c):
    if num < c:
        return 0
    else:
        return 1


def lbp_response(img):
    padded = pad(img)
    rows, cols = padded.shape
    lbp_img = np.zeros((rows - 2, cols - 2))
    for i in range(rows):
        for j in range(cols):
            if i < rows - 2 and j < cols - 2:
                i_c = padded[i + 1, j + 1]
                i_7 = return_i(padded[i, j], i_c) * (2 ** 0)
                i_6 = return_i(padded[i, j + 1], i_c) * (2 ** 1)
                i_5 = return_i(padded[i, j + 2], i_c) * (2 ** 2)
                i_4 = return_i(padded[i + 1, j + 2], i_c) * (2 ** 3)
                i_3 = return_i(padded[i + 2, j + 2], i_c) * (2 ** 4)
                i_2 = return_i(padded[i + 2, j + 1], i_c) * (2 ** 5)
                i_1 = return_i(padded[i + 2, j], i_c) * (2 ** 5)
                i_0 = return_i(padded[i + 1, j], i_c) * (2 ** 7)
                i_c = i_7 + i_6 + i_5 + i_4 + i_3 + i_2 + i_1 + i_0
                lbp_img[i, j] = round(i_c)

    plt.imshow(lbp_img)
    plt.title("LBP Response")
    plt.show()
    return lbp_img


def main():
    img = cv.imread('images/img2.jpg', 0)
    plt.imshow(img)
    plt.title("Input Image")
    plt.show()

    lbp_img = lbp_response(img)
    cv.imwrite('output/lbp_image.jpg', lbp_img)


if __name__ == "__main__":
    main()