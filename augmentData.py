from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os

datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='reflect')  # Use 'constant' fill mode with value 0

input_dir = '/content/drive/MyDrive/data'
output_dir = '/content/drive/MyDrive/dataAug'

for filename in os.listdir(input_dir):
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)  # Load the image
    mean_color = np.mean(img, axis=(0,1))  # Calculate the mean color of the image
    x = np.asarray(img)  # Convert image to numpy array
    x = x.reshape((1,) + x.shape)  # Reshape to (1, *image_shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=output_dir, save_prefix='aug', save_format='jpeg'):
        i += 1
        if i >= 10:  # Generate 10 augmented images from each original image
            break  # Stop the loop

        # Fill in the edges with the mean color
        # batch[batch==0] = mean_color
