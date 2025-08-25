from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Path to the saved model (.h5 file)
model_path = 'C:/Users/Vivek/OneDrive/Desktop/Solar Panel Research/GEN AI/generator.keras'

# Load the model
model = load_model(model_path)

# Print the model summary (to check its architecture)
model.summary()

# Generate a random input (example for GAN models)
noise = np.random.normal(0, 1, (1, 100))  # 1 sample, 100 dimensions (for example)

# Generate a prediction (in this case, a generated image from the GAN)
generated_image = model.predict(noise)

# Display the generated image (assuming the output is an image)
plt.imshow(generated_image[0])  # Assuming the output is a 64x64 image
plt.show()
