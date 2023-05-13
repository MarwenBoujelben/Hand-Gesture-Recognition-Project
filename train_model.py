#run this code on colab
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
# Load dataset
dataset_path = '/content/drive/MyDrive/allDataAug/'
class_names = os.listdir(dataset_path)
X, y = [], []
for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    for filename in os.listdir(class_path):
        img_path = os.path.join(class_path, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        X.append(img)
        y.append(class_names.index(class_name))
    print("class name:"+class_name)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Define model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))
# Evaluate model on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)


# Save the entire model to a file
model.save('/content/drive/MyDrive/model/modelFinal.h5')


