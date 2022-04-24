#!/usr/bin/env python3
# coding: utf-8

'''
Create a gallery of images based on a list of image IDs.

Dependencies:
`pip3 install numpy opencv-python`
'''

import sys
import os
import numpy as np
import cv2


class Montage:
    '''
    Combine input images into a single image displaying a combined grid of the input images.
    File minimally adapted from https://github.com/AdamSpannbauer/iphone_app_icon/blob/master/utils/resultsmontage.py
    '''
    def __init__(self, image_size, images_per_main_axis, num_results, by_row=True):
        # Store the target image size and the number of images per row.
        self.imageW = image_size[0]
        self.imageH = image_size[1]
        self.images_per_main_axis = images_per_main_axis
        self.by_row = by_row

        # Allocate memory for the output image.
        num_main_axis = -(-num_results // images_per_main_axis)  # ceiling division
        if by_row:
            self.montage = np.zeros(
                (num_main_axis * self.imageW, min(images_per_main_axis, num_results) * self.imageH, 3),
                dtype="uint8"
            )
        else:
            self.montage = np.zeros(
                (min(images_per_main_axis, num_results) * self.imageW, num_main_axis * self.imageH, 3),
                dtype="uint8"
            )

        # Initialize the counter for the current image along with the row and column number.
        self.counter = 0
        self.row = 0
        self.col = 0

    def add(self, image, text=None, highlight=False):
        # See if the number of images per row/col has been met, and if so,
        # reset the row/col counter and increment the row
        if self.by_row:
            if self.counter != 0 and self.counter % self.images_per_main_axis == 0:
                self.col = 0
                self.row += 1
        else:
            if self.counter != 0 and self.counter % self.images_per_main_axis == 0:
                self.col += 1
                self.row = 0

        # Resize image to the fixed width and height and set it in the montage.
        image = cv2.resize(image, (self.imageH, self.imageW))
        (startY, endY) = (self.row * self.imageW, (self.row + 1) * self.imageW)
        (startX, endX) = (self.col * self.imageH, (self.col + 1) * self.imageH)
        self.montage[startY:endY, startX:endX] = image

        # Check if some text should be added.
        if text is not None:
            o = 9
            cv2.putText(self.montage, text, (startX + o, startY + o*3), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(self.montage, text, (startX + o, startY + o*3), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 1, cv2.LINE_AA)

        # Check if the result should be highlighted.
        if highlight:
            b = 3
            cv2.rectangle(self.montage, (startX + b, startY + b), (endX - b, endY - b), (255, 255, 0), b*2)

        if self.by_row:
            self.col += 1
        else:
            self.row += 1

        self.counter += 1


def plot_gallery(query_img, ranked_imgs, outfile, img_size=(150,150)):
    '''
    Plot retrieval results as a gallery-like image file.
    '''
    numimgs = len(ranked_imgs) + 1 # include query
    gallery = Montage(image_size=img_size, images_per_main_axis=numimgs, num_results=numimgs, by_row=True)

    def load_img(img_path, hilite=False):
        result = cv2.imread(img_path)
        resize = cv2.resize(result, img_size, interpolation=cv2.INTER_AREA)
        gallery.add(resize, highlight=hilite)

    load_img(query_img, hilite=True)
    for image in ranked_imgs:
        load_img(image)

    # Ensure output dir exists, otherwise create it.
    outdir = os.path.dirname(outfile)
    #if not os.path.isdir(outdir):
    #    os.makedirs(outdir)

    cv2.imwrite(outfile, gallery.montage)
    print('Gallery saved as {}'.format(outfile), file=sys.stderr)


if __name__ == '__main__':
    # Demo: '1.jpg' is the query image and ['2.jpg', '3.jpg', '4.jpg'] is the top-3 retrieved images.
    plot_gallery('1.jpg', ['2.jpg', '3.jpg', '4.jpg'], 'test.png')
