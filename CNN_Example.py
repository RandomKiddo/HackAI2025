import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Define dataset paths
IMAGE_FOLDER = "data"
CSV_FILE = "iss.csv"

# Image parameters
IMG_SIZE = 512 
BATCH_SIZE = 16
EPOCHS = 75  # Adjust as needed

# Load CSV data
df = pd.read_csv(CSV_FILE)

# Parse (x, y) coordinates from string format "[x, y]"
df["x"] = df["location"].apply(lambda xy: int(xy.strip("[]").split(",")[0]))
df["y"] = df["location"].apply(lambda xy: int(xy.strip("[]").split(",")[1]))
df["depth"] = df["distance"]  # Rename for clarity

# Normalize labels (x, y to 0-1 range, depth as is)
IMG_WIDTH, IMG_HEIGHT = 512.0, 512.0  # Assuming original image size
df["x"] = df["x"] / IMG_WIDTH
df["y"] = df["y"] / IMG_HEIGHT
# df["depth"] = df["depth"] / df["depth"].max()  # Normalize depth

# Split into Train (80%), Test (15%), and Val (15%) sets
train_df, temp_df = train_test_split(df, test_size=0.3)     # 30% for temp (val + test)
val_df, test_df = train_test_split(temp_df, test_size=0.5)  # Split temp in half

# Load and preprocess images
def load_image(imageid):
    image_path = os.path.join(IMAGE_FOLDER, str(imageid))  # Ensure string format
    if not os.path.exists(image_path):                     # Check if file exists
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:  # Check if image loading failed
        raise ValueError(f"Failed to load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize pixel values
    return img

# Create dataset
def data_generator(dataframe):
    for _, row in dataframe.iterrows():
        image = load_image(str(row["ImageID"])+".jpg")
        label = np.array([
            row["x"], 
            row["y"], 
            # row["depth"]
            ], dtype=np.float32
        )
        yield image, label

# Convert to TensorFlow datasets
# Train Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_df),
    output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.float32)
    )
).batch(BATCH_SIZE).shuffle(1000).prefetch(tf.data.AUTOTUNE)

# Validation Dataset
val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_df),
    output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.float32)
    )
).batch(BATCH_SIZE).shuffle(1000).prefetch(tf.data.AUTOTUNE)

# Test Dataset
test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(test_df),
    output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(2,), dtype=tf.float32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # No shuffle for test set

# Define custom depth accuracy metric --> we want this to be close to 0
@tf.keras.utils.register_keras_serializable()
def depth_difference(y_true, y_pred):
    # Extract depth values (last column)
    y_true_depth = y_true[:, 2]
    y_pred_depth = y_pred[:, 2]

    # Compute absolute difference
    return tf.reduce_mean(tf.abs(y_true_depth - y_pred_depth))

# Build lightweight CNN model
base_model = tf.keras.applications.MobileNetV3Large(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base model for faster training

# Unfreeze last 10 layers
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Data Augmentation Layer
data_augmentation = tf.keras.Sequential([ 
    tf.keras.layers.RandomBrightness(0.05),  # Adjust brightness by ±5%
    tf.keras.layers.RandomContrast(0.05)     # Adjust contrast by ±5%
])

model = tf.keras.Sequential([
    # data_augmentation,
    base_model,                                                       # Extract spatial features
    # tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),            # Conv2D layer, kernel size 3, 3
    # tf.keras.layers.Dropout(0.3),                                     # Dropout to help with overfitting
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),            # Conv2D layer, kernel size 3, 3
    # tf.keras.layers.BatchNormalization(),                             # Normalization to transition from 'relu' to 'swish'
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # SeparableConv2D layer, kernel size 3, 3
    tf.keras.layers.Flatten(),                                        # Preserve all spatial details
    # tf.keras.layers.Dropout(0.3),                                     # Dropout to help with overfitting
    tf.keras.layers.Dense(16, activation="relu"),                    # Dense Layer
    tf.keras.layers.Dense(2, activation="linear")                    # Output: (x, y)
])

# Compile model with the custom metric correctly referenced
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), 
    loss="mse",
    metrics=[
        "mae", 
        "mse", 
        # depth_difference
    ]
)

# print(base_model.summary())

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',         # Metric to monitor
    patience=10,                # Number of epochs to wait for improvement
    restore_best_weights=True,  # Restore the best weights when stopping
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "keypoint_model7.keras", 
    monitor="val_loss", 
    mode="min", 
    save_best_only=True, 
    verbose=1
)

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",  # Reduce when val_loss stops improving
    factor=0.5,         # Halve learning rate
    patience=2,          # Wait 2 epochs before reducing
    min_lr=5e-8,         # Minimum learning rate
    min_delta=1e-5,      # Minimum delta needed to trigger ReduceLROnPlateau()
    verbose=1
)

# Train model
history =  model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stopping, 
        checkpoint, 
        lr_schedule
        ],
    verbose=1
)

print("Model training complete! 🎯")

# Plot validation loss vs loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Validation Loss vs Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('loss_plot7.png')
plt.show()

best_model = tf.keras.models.load_model(
    'keypoint_model7.keras',
    # custom_objects={"depth_difference": depth_difference}
)
best_model.evaluate(test_dataset)

print("Model testing complete! 🎯")
print(best_model.summary())