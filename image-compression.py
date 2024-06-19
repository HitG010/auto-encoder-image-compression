# AutoEncoders for Image Compression
# What is an AutoEncoder?
# An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for the purpose of dimensionality reduction. Also, the autoencoder can be used to reduce the noise in the data. The autoencoder is trained to map the input to the output, and the output to the input. The autoencoder learns to ignore noise in the data.x
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.datasets import mnist

# Load the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data() # We don't need the labels as autoencoders are unsupervised

# Normalize the data
x_train = x_train.astype('float32') / 255   
x_test = x_test.astype('float32') / 255

# Reshape the data
x_train = x_train.reshape((len(x_train), 28, 28, 1)) # 1 is the number of channels as the images are grayscale
x_test = x_test.reshape((len(x_test), 28, 28, 1))    
print(x_train.shape) # (60000, 28, 28, 1)

# Data visualization
index = np.random.randint(len(x_test))
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title('Original Image')
plt.show()
# Create the autoencoder model
model = Sequential([
    # Encoder network
    Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D(2, padding='same'),
    Conv2D(16, 3, activation='relu', padding='same'),
    MaxPooling2D(2, padding='same'),
    # Decoder network
    Conv2D(16, 3, activation='relu', padding='same'),
    UpSampling2D(2),
    Conv2D(32, 3, activation='relu', padding='same'),
    UpSampling2D(2),
    # Output layer
    Conv2D(1, 3, activation='sigmoid', padding='same')
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

num_epochs = 10
# Train the model
model.fit(x_train, x_train, epochs=num_epochs, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Save the model
model.save(f'autoencoder_epochs={num_epochs}.keras')

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
