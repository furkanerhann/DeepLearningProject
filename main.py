import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

model = keras.models.load_model('mymodel.h5')

#Image
image_address = "input01.jpg"

image = cv2.imread(image_address, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY_INV)
image2pred = cv2.resize(th, (28, 28))
input_image = np.reshape(image2pred, (1, 784)).astype('float32') / 255
prediction = model.predict(input_image)
predicted_label = np.argmax(prediction)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(image, cmap=plt.cm.binary)
axs[0].set_title("Input")
axs[0].axis('off')
axs[1].imshow(image2pred, cmap=plt.cm.binary)
axs[1].set_title(f"Predicted: {predicted_label}")
axs[1].axis('off')
plt.show()