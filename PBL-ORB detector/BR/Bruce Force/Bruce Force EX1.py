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
 
# main function
if __name__ == '__main__':
 # giving the path of both of the images
    first_image_path = 'Input/test0.jpg'
    second_image_path = 'Input/test1.jpg'
 
    # reading the image from there path by calling the function
    img1, img2 = read_image(first_image_path,second_image_path)
 
    # converting the readed images into the gray scale images by calling the function
    gray_pic1, gray_pic2 = convert_to_grayscale(img1,img2)
    cv2.imshow('Gray scaled image 1',gray_pic1)
    cv2.imshow('Gray scaled image 2',gray_pic2)
    cv2.waitKey()
    cv2.destroyAllWindows()