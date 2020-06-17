import matplotlib.pyplot as plt
import cv2 as cv
import operations
import argparse

img = None

#plt.imshow(img)
#plt.show()


class Switcher(object):
    def operation(self, i):
        method_name = 'operation_' + i
        method = getattr(self, method_name, lambda: 'Invalid')
        return method()

    def operation_0(self):
        grey_img = operations.convert_to_greyscale(img)
        status = cv.imwrite('output/new_grey.png', grey_img)
        if status:
            plt.imshow(grey_img)
            plt.title("Greyscale Image")
            plt.show()

    def operation_1(self):
        operations.plot_histogram(img)

    def operation_2(self):
        operations.equalize(img)

    def operation_3(self):
        neg_img = operations.negative_image(img)
        status = cv.imwrite('output/neg_img.png', neg_img)
        if status:
            plt.imshow(neg_img)
            plt.title("Negative image")
            plt.show()
        else:
            print("error while saving image")

    def operation_4(self):
        c = float(input("Enter constant value (default = 20)"))
        if c is None:
            c = 25
        log_img = operations.log_transform(img, 25)
        status = cv.imwrite('output/log_trans_img.png', log_img)
        if status:
            plt.imshow(log_img)
            plt.title("Log Transformation with c = {}".format(c))
            plt.show()
        else:
            print("error while saving image")

    def operation_5(self):
        g = float(input("enter value of gamma (c = 1 (default))"))
        pow_img = operations.power_transform(img, 1, g)
        status = cv.imwrite('output/pow_trans_img.png', pow_img)
        if status:
            plt.imshow(img)
            plt.title("Power Law Transformation with c = 1 and gamma = {}".format(g))
            plt.show()
        else:
            print("error while saving image")

    def operation_6(self):
        stretched = operations.contrast_stretch(img)
        status = cv.imwrite('output/contrast_stretch.png', stretched)
        if status:
            plt.imshow(img)
            plt.show()
            plt.imshow(stretched)
            plt.show()
        else:
            print("error while saving image")

    def operation_7(self):
        for bit_plane in range(8):
            sliced = operations.bit_plane_slice(img, bit_plane)
            status = cv.imwrite('output/bit_plane_slice/bit_plane_{args}.png'.format(args=bit_plane), sliced)
            if status:
                plt.imshow(sliced)
                plt.title('Bit plane {}'.format(bit_plane))
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
        global img
        img = cv.imread(file_path)
    switch = Switcher()
    print("Press 0 to convert image to greyscale\nPress 1 to plot histogram\nPress 2 to equalize histogram\n" +
          "Press 3 to get negative of image\nPress 4 for Log Transformation\nPress 5 for Power Law Transformation\n"
          + "Press 6 for Contrast Stretching\nPress 7 for Bit plane slicing of 8 bit image\n")

    x = input()
    switch.operation(x)


if __name__ == "__main__":
    main()
