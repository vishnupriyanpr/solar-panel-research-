# cnn_gpu_enabled.py (Rewritten to enforce GPU-only training and full pipeline compliance)

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# ✅ Force TensorFlow to use GPU and throw error if not available
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError("❌ No GPU found. Please check CUDA installation and GPU availability.")
else:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"✅ GPU detected and configured: {gpus[0].name}")
    except RuntimeError as e:
        raise RuntimeError(f"❌ GPU initialization error: {e}")

# ✅ Paths to Dataset
train_dir = 'D:/Solar Panel Research/sorted_images/train'
val_dir = 'D:/Solar Panel Research/sorted_images/validation'
test_dir = 'D:/Solar Panel Research/sorted_images/test'

# ✅ Image Parameters
img_height, img_width = 224, 224
batch_size = 32
epochs = 15

# ✅ Data Augmentation and Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ✅ CNN Model Definition
model = models.Sequential([
    layers.InputLayer(input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# ✅ Compile Model
optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train the Model (on GPU)
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# ✅ Evaluate on Test Data (on GPU)
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"🧪 Test Accuracy: {test_accuracy * 100:.2f}%")

# ✅ Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

# ✅ Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()
