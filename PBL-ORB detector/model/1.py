import cv2
import numpy as np

def detector(image1,image2):
    # creating ORB detector
    detect = cv2.ORB_create()
 
    # finding key points and descriptors of both images using
    # detectAndCompute() function
    key_point1,descrip1 = detect.detectAndCompute(image1,None)
    key_point2,descrip2 = detect.detectAndCompute(image2,None)
    return (key_point1,descrip1,key_point2,descrip2)
 
def BF_FeatureMatcher(des1,des2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    no_of_matches = brute_force.match(des1,des2)
 
    # finding the humming distance of the matches and sorting them
    no_of_matches = sorted(no_of_matches,key=lambda x:x.distance)
    return no_of_matches

def display_output(pic1,kpt1,pic2,kpt2,best_match):
 
    # drawing the feature matches using drawMatches() function
    output_image = cv2.drawMatches(pic1,kpt1,pic2,kpt2,best_match,None,flags=2)
    cv2.imwrite('O:/BR/Output 3/output_image.jpg',output_image)
    cv2.imshow('Output image',output_image)
    
 
# Read Image to be aligned
imgTest = cv2.imread('Input/test2.jpg')
# Reference Reference image or Ideal image
imgRef = cv2.imread('Input/test3.jpg')
imgTest_grey = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
imgRef_grey = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
height, width = imgRef_grey.shape
# Configure ORB feature detector Algorithm with 1000 features.
orb_detector = cv2.ORB_create(1000)
 
# Extract key points and descriptors for both images
keyPoint1, des1 = orb_detector.detectAndCompute(imgTest_grey, None)
keyPoint2, des2 = orb_detector.detectAndCompute(imgRef_grey, None)
 
# Display keypoints for reference image in green color
imgKp_Ref = cv2.drawKeypoints(imgRef, keyPoint2, 0, (0,222,0), None)
imgKp_Ref = cv2.resize(imgKp_Ref, (width//1, height//1))

imgKp_Test = cv2.drawKeypoints(imgTest, keyPoint1, 0, (0,222,0), None)
imgKp_Test = cv2.resize(imgKp_Test, (width//1, height//1))
cv2.imshow('Key Points', imgKp_Ref)
cv2.imshow('Key Point', imgKp_Test)
cv2.waitKey(0)


key_pt1,descrip1,key_pt2,descrip2 = detector(imgTest,imgKp_Ref)
number_of_matches = BF_FeatureMatcher(descrip1,descrip2)
tot_feature_matches = len(number_of_matches)
 
    # printing total number of feature matches found
print(f'Total Number of Features matches found are {tot_feature_matches}')
 
    # after drawing the feature matches displaying the output image
display_output(imgTest,key_pt1,imgRef,key_pt2,number_of_matches)
cv2.waitKey()
cv2.destroyAllWindows()