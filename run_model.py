import cv2
import numpy as np
import tensorflow as tf
from google.colab.patches import cv2_imshow #use this when running the code on colab because cv2.imshow is not recognized in colab

# Load the entire model from a file
model = tf.keras.models.load_model('/content/drive/MyDrive/model/modelFinal.h5')

# Load class names
class_names = ['Channel DOWN', 'Channel UP', 'Volume DOWN', 'Volume UP', 'Power ON', 'Power OFF']

# Load the image from drive
image_path = '/content/drive/MyDrive/allDataAug/PowerOn/aug_0_574.jpeg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Preprocess the image
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=-1)
img = np.repeat(img, 3, axis=-1)

# Predict the image
pred = model.predict(img)[0]
class_name = class_names[np.argmax(pred)]

# Display the prediction on the image
image_with_prediction = cv2.imread(image_path)
cv2.putText(image_with_prediction, class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the image with prediction
cv2_imshow(image_with_prediction)
#use the following commented line while running locally 
#cv2.imshow('image with prediction',image_with_prediction)
cv2.waitKey(0)
cv2.destroyAllWindows()
