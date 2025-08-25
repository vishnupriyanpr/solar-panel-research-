import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Path to the saved discriminator model
discriminator_path = 'C:/Users/Vivek/OneDrive/Desktop/Solar Panel Research/GEN AI/discriminator.keras'

# Load the saved discriminator model
discriminator = tf.keras.models.load_model(discriminator_path)

# Show the model summary (architecture)
discriminator.summary()

# Load the generator model to generate images
generator_path = 'C:/Users/Vivek/OneDrive/Desktop/Solar Panel Research/GEN AI/generator.keras'
generator = tf.keras.models.load_model(generator_path)

# Generate a random input (noise vector for generator)
noise = np.random.normal(0, 1, (1, 100))  # 1 sample, 100 dimensions (for example)

# Generate a prediction (generate an image using the generator)
generated_image = generator.predict(noise)

# Now, pass the generated image through the discriminator
discriminator_output = discriminator.predict(generated_image)

# Display the generated image
plt.imshow(generated_image[0])  # Assuming the output is a 64x64 image
plt.title(f'Discriminator Output: {discriminator_output[0]}')
plt.show()
