# -*- coding: utf-8 -*-


import os
os.chdir(r"C:/Users/CSE/Desktop/Bijoyeta/FBIM")
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from BFfunction import imagereg
from scipy import ndimage as ndi
# from skimage import (exposure, feature, filters, io, measure, morphology, restoration, segmentation, transform, util)
import napari



os.chdir(r"C:/Users/CSE/Desktop/Bijoyeta/FBIM")


""" Image Registration and 3D Model """
x = 1
images = list()
while(True):
    
    y = str(x)
    img1_color = cv2.imread('out/out'+y+'.jpg')  # Image to be aligned.
    img2_color = cv2.imread('input/before.jpg')    # Reference image.
    
    transformed_img = imagereg(img1_color,img2_color) # Image registration
    
    # print('test'+y+'.jpg')  # Show non aligned image
    # plt.imshow(img1_color)
    # plt.show()
    
    print('out'+y+'transformed.jpg')
    cv2.imwrite('output/t'+y+'.jpg',transformed_img)   # Show aligned image
    print("Saved")
    plt.imshow(transformed_img)
    plt.show()
    
    
    images.append(transformed_img)  # Generating 2D image list
    
    x =x+1
    
    if(x>10):
        break

