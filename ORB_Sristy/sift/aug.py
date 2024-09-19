from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io

# Initialize the ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.0,
    height_shift_range=0.0,
    shear_range=0.2,
    zoom_range=(0.5, 1.0),  # Valid zoom range
    horizontal_flip=False,
    fill_mode='constant'
)

# Load the image
try:
    x = io.imread("/home/sristy/Desktop/ORB-Descriptor/ORB_Sristy/sift/Colon/10x/1_colon_10x.tif")
except Exception as e:
    print(f"Error loading image: {e}")

# Ensure the image is properly shaped
if x.ndim == 2:  # Grayscale image
    x = x.reshape((1, x.shape[0], x.shape[1], 1))  # Reshape to (1, height, width, channels)
elif x.ndim == 3:  # RGB or RGBA image
    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))  # Reshape to (1, height, width, channels)
else:
    raise ValueError("Unexpected image shape: {}".format(x.shape))

# Generate augmented images
output_dir = "/home/sristy/Desktop/ORB-Descriptor/ORB_Sristy/sift/Colon/output_colon"
i = 0
for batch in datagen.flow(x, batch_size=16,
                          save_to_dir=output_dir,
                          save_prefix='output_img',
                          save_format='jpg'):
    i += 1
    if i > 10:  # Stop after generating 10 images
        break
