import cv2
import numpy as np
import napari
import os

# Folder containing the JPEG images
folder = 'output'

# List of filenames in the folder
filenames = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]

# Load the images into a list
imgs = [cv2.imread(f) for f in filenames]

# Convert the list of images into a 3D numpy array
img_array = np.stack(imgs, axis=0)

# View the 3D image using napari
with napari.gui_qt():
    viewer = napari.view_image(img_array)
