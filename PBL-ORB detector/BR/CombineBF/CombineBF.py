# Author: Om Sinkar

import cv2
 
# function to read the images by taking there path
def read_image(path1,path2):
  # reading the images from their using imread() function
    read_img1 = cv2.imread(path1)
    read_img2 = cv2.imread(path2)
    return (read_img1,read_img2)
 
# function to convert images from RGB to gray scale
def convert_to_grayscale(pic1,pic2):
    gray_img1 = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(pic2,cv2.COLOR_BGR2GRAY)
    return (gray_img1,gray_img2)
 
def detector(image1,image2):
	# creating ORB detector
	detect = cv2.ORB_create()

	# finding key points and descriptors of both images using detectAndCompute() function
	key_point1,descrip1 = detect.detectAndCompute(image1,None)
	key_point2,descrip2 = detect.detectAndCompute(image2,None)
	return (key_point1,descrip1,key_point2,descrip2)

def BF_FeatureMatcher(des1,des2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    no_of_matches = brute_force.match(des1,des2)
 

    no_of_matches = sorted(no_of_matches,key=lambda x:x.distance)
    return no_of_matches

def BF_FeatureMatcher(des1,des2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    no_of_matches = brute_force.match(des1,des2)
 
    # finding the humming distance of the matches and sorting them
    no_of_matches = sorted(no_of_matches,key=lambda x:x.distance)
    return no_of_matches
 
# function displaying the output image with the feature matching
# def display_output(pic1,kpt1,pic2,kpt2,best_match):
#     # drawing the feature matches using drawMatches() function
#     output_image = cv2.drawMatches(pic1,kpt1,pic2,kpt2,best_match[:30],None,flags=2)
#     cv2.imshow('Output image',output_image)
 

def display_output(pic1,kpt1,pic2,kpt2,best_match):
 

    output_image = cv2.drawMatches(pic1,kpt1,pic2,kpt2,best_match,None,flags=2)
    cv2.imwrite('Output 3/output_image.jpg',output_image)
    cv2.imshow('Output image',output_image)
    
# main function
if __name__ == '__main__':
 # giving the path of both of the images
    first_image_path = 'Input/test0.jpg'
    second_image_path = 'Input/test1.jpg'
 
    # reading the image from there path by calling the function
    img1, img2 = read_image(first_image_path,second_image_path)
 
    # converting the readed images into the gray scale images by calling the function
    cv2.imshow('Gray scaled image 1',img1)
    cv2.imshow('Gray scaled image 2',img2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    key_pt1,descrip1,key_pt2,descrip2 = detector(img1,img2)
    cv2.imshow("Key points of Image 1",cv2.drawKeypoints(img1,key_pt1,None))
    cv2.imshow("Key points of Image 2",cv2.drawKeypoints(img2,key_pt2,None))
    out1 = cv2.drawKeypoints(img1,key_pt1,None)
    out2 = cv2.drawKeypoints(img2,key_pt2,None)
    cv2.imwrite('Output/out1.jpg',out1)
    cv2.imwrite('Output/out2.jpg',out2)
    # printing descriptors of both of the images
    print(f'Descriptors of Image 1 {descrip1}')
    print(f'Descriptors of Image 2 {descrip2}')
    print('------------------------------')
    cv2.imwrite('Output/out1_1.jpg',out1)
    cv2.imwrite('Output/out2_1.jpg',out2)
	# printing the Shape of the descriptors
    print(f'Shape of descriptor of first image {descrip1.shape}')
    print(f'Shape of descriptor of second image {descrip2.shape}')
    cv2.waitKey()
    cv2.destroyAllWindows()

    number_of_matches = BF_FeatureMatcher(descrip1,descrip2)
    tot_feature_matches = len(number_of_matches)
 
    # printing total number of feature matches found
    print(f'Total Number of Features matches found are {tot_feature_matches}')
 
    # after drawing the feature matches displaying the output image
    display_output(img1,key_pt1,img2,key_pt2,number_of_matches)
    cv2.waitKey()
    cv2.destroyAllWindows()