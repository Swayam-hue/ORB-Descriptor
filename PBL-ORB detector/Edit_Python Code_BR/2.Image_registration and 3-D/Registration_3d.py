# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:54:16 2022

@author: Pallav Bishi
"""

""" Importing Libraries """

import os
os.chdir(r'O:/Edit_Python Code_BR/2.Image_registration and 3-D')
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from registrationlib import imagereg
from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
import napari
from PIL import Image as im

""" Changing Dir"""

os.chdir("O:/Edit_Python Code_BR/2.Image_registration and 3-D/input")


""" Image Registration and 3D Model """
x = 0
images = list()
while(True):
    
    y = str(x)
    img1_color = cv2.imread('test'+y+'.jpg')  # Image to be aligned.
    img2_color = cv2.imread('before.jpg')    # Reference image.
    
    transformed_img = imagereg(img1_color,img2_color) # Image registration
    
    print('test'+y+'.jpg')  # Show non aligned image
    plt.imshow(img1_color)
    plt.show()
    
    print('test'+y+'transformed.jpg')   # Show aligned image
    plt.imshow(transformed_img)
    plt.show()
    
    
    images.append(transformed_img)  # Generating 2D image list
    
    x =x+1
    
    if(x>4):
        break

""" View 3D model """

image_array = np.array(images)
print(image_array)
viewer = napari.view_image(image_array,rgb=True)


