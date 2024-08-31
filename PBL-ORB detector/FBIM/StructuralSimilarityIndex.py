from skimage.metrics import structural_similarity as ssim
import cv2

# Load the images
img1 = cv2.imread('input/before.jpg')
img2 = cv2.imread('output/t10.jpg')
# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Compute the SSIM between the images
ssim_value = ssim(gray1, gray2)

# Print the SSIM value
print('Structural Similarity Index:', ssim_value)
