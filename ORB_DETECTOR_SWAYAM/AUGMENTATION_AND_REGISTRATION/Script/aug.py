from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np



datagen = ImageDataGenerator(
    rotation_range=30,  # Adjust rotation range as needed
    width_shift_range=0,  # Adjust shifting parameters as needed
    height_shift_range=0,
    shear_range=0,
    zoom_range=-0.4,
    horizontal_flip=False,
    fill_mode='constant',
    cval = 0
)

y = cv2.imread('Augmentation_2/input/before.jpg')
y = y.reshape((1,) + y.shape)

# Specify the desired window size
  # Adjust as needed

i = 0
for batch in datagen.flow(y, batch_size=16,
                         save_to_dir='Augmentation_2/augmented_4',
                         save_prefix='aug',
                         save_format='jpg'):
    for image in batch:
        cv2.imwrite('Augmentation_2/augmented_4/aug_' + str(i) + '.jpg', image)
    i += 1
    if i > 5:
        break