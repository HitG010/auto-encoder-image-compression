import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.datasets import mnist

# Load the model
model = load_model('autoencoder_epochs=10.keras')

# Load the MNIST dataset
(_, _), (x_test, _) = mnist.load_data()

# Normalize the data
x_test = x_test.astype('float32') / 255

# Reshape the data
x_test = x_test.reshape((len(x_test), 28, 28, 1))

# Data visualization
index = np.random.randint(len(x_test))
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title('Original Image')
plt.show()

# Predict the output
output = model.predict(x_test[index].reshape(1, 28, 28, 1))

# Data visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(output.reshape(28, 28), cmap='gray')
plt.title('Reconstructed Image')
plt.show()
