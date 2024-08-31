from skimage import io
from image_registration import chi2_shift

image = io.imread("images/BSE.jpg")
offset_image = io.imread("images/BSE_transl.jpg")
# offset image translated by (-17.45, 18.75) in y and x 

#Method 1: chi squared shift
#Find the offsets between image 1 and image 2 using the DFT upsampling method
noise=0.1
xoff, yoff, exoff, eyoff = chi2_shift(image, offset_image, noise, return_error=True, upsample_factor='auto')

print("Offset image was translated by: 18.75, -17.45")
print("Pixels shifted by: ", xoff, yoff)

from scipy.ndimage import shift
corrected_image = shift(offset_image, shift=(xoff,yoff), mode='constant')

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected')
plt.show()