# importing openCV library
import cv2

# function to read the images by taking there path
def read_image(path1,path2):
	read_img1 = cv2.imread(path1)
	read_img2 = cv2.imread(path2)
	return (read_img1,read_img2)

# function to convert images from RGB to gray scale
def convert_to_grayscale(pic1,pic2):
	gray_img1 = cv2.cvtColor(pic1,cv2.COLOR_BGR2GRAY)
	gray_img2 = cv2.cvtColor(pic2,cv2.COLOR_BGR2GRAY)
	return (gray_img1,gray_img2)

# function to detect the features by finding key points and descriptors from the image
def detector(image1,image2):
	# creating ORB detector
	detect = cv2.ORB_create()

	# finding key points and descriptors of both images using detectAndCompute() function
	key_point1,descrip1 = detect.detectAndCompute(image1,None)
	key_point2,descrip2 = detect.detectAndCompute(image2,None)
	return (key_point1,descrip1,key_point2,descrip2)

# main function
if __name__ == '__main__':
# giving the path of both of the images
	first_image_path = "Input/test0.jpg"
	second_image_path = "Input/test1.jpg"
	img1, img2 = read_image(first_image_path,second_image_path)

	# converting the readed images into the gray scale images
	gray_pic1, gray_pic2 = convert_to_grayscale(img1,img2)

	# storing the finded key points and descriptors of both of the images
	key_pt1,descrip1,key_pt2,descrip2 = detector(img1,img2)

	# showing the images with their key points finded by the detector
	cv2.imshow("Key points of Image 1",cv2.drawKeypoints(img1,key_pt1,None))
	cv2.imshow("Key points of Image 2",cv2.drawKeypoints(img2,key_pt2,None))
	out1 = cv2.drawKeypoints(img1,key_pt1,None)
	out2 = cv2.drawKeypoints(img2,key_pt2,None)
	cv2.imwrite('O:/BR/Output 2/out1.jpg',out1)
	cv2.imwrite('O:/BR/Output 2/out2.jpg',out2)

	# printing descriptors of both of the images
	print(f'Descriptors of Image 1 {descrip1}')
	print(f'Descriptors of Image 2 {descrip2}')
	print('------------------------------')

	# printing the Shape of the descriptors
	print(f'Shape of descriptor of first image {descrip1.shape}')
	print(f'Shape of descriptor of second image {descrip2.shape}')

	cv2.waitKey()
	cv2.destroyAllWindows()
