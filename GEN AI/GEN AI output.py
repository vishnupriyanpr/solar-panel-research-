import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from sklearn.model_selection import train_test_split
import shutil

# Set image path
image_path = r'C:\Users\Vivek\OneDrive\Desktop\Solar Panel Research\sorted_images'

# Load and preprocess images
def load_data(image_path, img_size=(64, 64)):
    images = []
    for root, _, files in os.walk(image_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                img = tf.keras.preprocessing.image.load_img(os.path.join(root, file), target_size=img_size)
                img = tf.keras.preprocessing.image.img_to_array(img) / 127.5 - 1.0
                images.append(img)
    return np.array(images)

dataset = load_data(image_path)
print(f'Loaded dataset with shape: {dataset.shape}')

# Split dataset into training and validation sets
train_data, val_data = train_test_split(dataset, test_size=0.2)

# Generator model
def build_generator():
    model = Sequential([
        Dense(8 * 8 * 256, input_dim=100),
        LeakyReLU(0.2),
        Reshape((8, 8, 256)),
        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        LeakyReLU(0.2),
        Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
        LeakyReLU(0.2),
        Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh')
    ])
    return model

# Discriminator model
def build_discriminator():
    model = Sequential([
        Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=(64, 64, 3)),
        LeakyReLU(0.2),
        Dropout(0.3),
        Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
        LeakyReLU(0.2),
        Dropout(0.3),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

# Save generated images to both generated_images and sorted folders
def save_generated_images(epoch, noise):
    generated_images = generator.predict(noise)
    folder = 'generated_images'
    os.makedirs(folder, exist_ok=True)
    categories = ['cell', 'cell multi', 'diode', 'diode multi', 'cracking', 'hot-spot', 'hot-spot multi', 'no-anomaly', 'offline module', 'shadowing', 'soiling', 'vegetation']
    subfolders = ['train', 'validation', 'test']

    for i, img in enumerate((generated_images + 1.0) * 127.5):
        file_name = f'generated_{epoch}_{i}.png'
        file_path = os.path.join(folder, file_name)
        plt.imsave(file_path, img.astype(np.uint8))

        # Sort image into correct folder
        category = np.random.choice(categories)
        subfolder = np.random.choice(subfolders)
        sorted_path = os.path.join(image_path, subfolder, category)
        os.makedirs(sorted_path, exist_ok=True)
        shutil.copy(file_path, os.path.join(sorted_path, file_name))

# Train GAN
def train_gan(epochs=100, batch_size=16):
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, 100))
        if epoch % 20 == 0:
            save_generated_images(epoch, noise)
        print(f'Epoch {epoch} completed.')

# Instantiate models
generator = build_generator()
discriminator = build_discriminator()
train_gan()
