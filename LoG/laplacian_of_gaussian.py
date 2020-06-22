import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math
import argparse

# Laplacian filters
laplace_1 = [
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
]

laplace_2 = [
    [-1, 2, -1],
    [2, -4, 2],
    [-1, 2, -1]
]


def create_gaussian_filter(x, y, sigma):
    gaussian_filter = np.zeros((x + x, y + x))
    total = 0
    for i in range(-x, x):
        for j in range(-y, y):
            a = (i ** 2) + (j ** 2)
            b = 2 * (sigma ** 2)
            gaussian_filter[i + 3, j + 3] = np.exp(-((a * a) / b)) / math.pi * b
            total = total + gaussian_filter[i + 3, j + 3]

    gaussian_filter = (gaussian_filter[:] / total)

    return gaussian_filter


def pad(img, padding):
    rows, cols, channels = img.shape
    padded = np.zeros((rows + padding, cols + padding, channels))

    for i in range(rows + padding):
        for j in range(cols + padding):
            for c in range(channels):
                if i != 0 and i != rows + padding - 1 and j != 0 and j != cols + padding - 1:
                    padded[i, j, c] = img[i - padding - 1, j - padding - 1, c]

    return padded


def apply_gaussian_filter(img, sigma):
    padded = pad(img, 4)
    rows, cols, channels = padded.shape
    blurred = np.zeros((rows - 6, cols - 6, channels))

    gaussian = create_gaussian_filter(3, 3, sigma)

    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                pixel = 0
                if i < rows - 6 and j < cols - 6:
                    for k in range(6):
                        for l in range(6):
                            pixel = pixel + (gaussian[k, l] * padded[i + k, j + l, c])
                    blurred[i, j, c] = int(round(pixel))

    return blurred


def detect_edge(img, kernel, sigma, out_path):
    blurred = apply_gaussian_filter(img, sigma)
    blur_padded = pad(blurred, 2)
    rows, cols, channels = blur_padded.shape
    x = len(kernel)
    y = len(kernel)

    edge_det = np.zeros((rows - 2, cols - 2, channels))

    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                pixel = 0
                if i != 0 and i < rows - 2 and j != 0 and j < cols - 2:
                    for k in range(3):
                        for l in range(3):
                            pixel = pixel + (kernel[k][l] * blur_padded[i + k, j + l, c])
                    edge_det[i, j, c] = int(round(pixel))

    status = cv.imwrite(out_path, edge_det)
    if status:
        print("File Saved at {}".format(out_path))
    else:
        print("Error While saving file")

    plt.imshow(edge_det)
    plt.title("LoG of image with sigma = {}".format(sigma))
    plt.xlabel("Height Pixels")
    plt.ylabel("Width Pixels")
    plt.show()


def process_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help="Input image path")
    return parser.parse_args()


def main():
    args = process_arguments()
    file_path = args.file

    if file_path is None:
        print("Use command python main.py --file images/img.png")
        return
    else:
        img = cv.imread(file_path)

    if img is not None:
        plt.imshow(img)
        plt.title("Original Image")
        plt.xlabel("Width Pixels")
        plt.ylabel("Height Pixels")
        plt.show()

        detect_edge(img, laplace_1, 1, 'output/log_kernel_1_sigma_1.jpg')

        detect_edge(img, laplace_1, 3, 'output/log_kernel_2_sigma_3.jpg')
    else:
        print("File not found.")


if __name__ == "__main__":
    main()
