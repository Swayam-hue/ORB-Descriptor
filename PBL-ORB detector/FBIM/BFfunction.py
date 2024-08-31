# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt 
os.chdir(r"C:/Users/CSE/Desktop/Bijoyeta/FBIM/out")

def imagereg(imgalign,refimg):

    img1_color = imgalign  # Image to be aligned.
    img2_color = refimg    # Reference image.
    cv2.imshow('Gray scaled image 1',img1_color)
    cv2.imshow('Gray scaled image 2',img2_color)
# 1. Convert both images to grayscale
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray scaled image 1',img1)
    cv2.imshow('Gray scaled image 2',img2)
    height, width = img2.shape

# 2. Fetch Keypoints and descriptor
    orb_detector = cv2.ORB_create(5000)

    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
    cv2.imshow("Key points of Image 1",cv2.drawKeypoints(img1,kp1,None))
    cv2.imshow("Key points of Image 2",cv2.drawKeypoints(img2,kp2,None))


# 3. Match the keypoints between two images using bruteforce matcher.
    # Hamming distance as measurement mode.
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
     
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
     
    no_of_matches = len(matches)
     
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    

     
    for i in range(len(matches)):
      p1[i, :] = kp1[matches[i].queryIdx].pt
      p2[i, :] = kp2[matches[i].trainIdx].pt
    

     
# 4. Find the homography matrix.

    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
     
# 5. Use this matrix to transform the
    # colored image wrt the reference image.
    
    transformed_img = cv2.warpPerspective(img1_color,homography, (width, height))
    return transformed_img


    