import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# ✅ Update these to match your dataset location
train_dir = r'C:\Users\MetaVerse KAHE\Desktop\ret\dataset\train'
val_dir = r'C:\Users\MetaVerse KAHE\Desktop\ret\dataset\val'


# ✅ Check if paths exist
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Train directory not found: {train_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation directory not found: {val_dir}")

# ✅ Training constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 100
FINE_TUNE_EPOCHS = 50
LEARNING_RATE = 1e-3

# ✅ Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.15,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ✅ Optional: Show class indices
print("Class indices:", train_generator.class_indices)

# ✅ Load base model
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# ✅ Build model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# ✅ Compile model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ✅ Initial training
print("🔧 Starting initial training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# ✅ Fine-tuning
print("🔧 Starting fine-tuning...")
base_model.trainable = True

# Freeze all layers except the last 20 in base model
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Re-compile
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Continue training
total_epochs = EPOCHS + FINE_TUNE_EPOCHS
history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=val_generator
)

# ✅ Save model
model.save('rp_detection_model.h5')
print("✅ Training complete. Model saved as 'rp_detection_model.h5'")
