# %%
import os
import random
import json
import pickle

import tensorflow as tf
import kagglehub as kh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
path = kh.dataset_download("msafi04/iss-docking-dataset")

# %%
df = pd.read_csv(os.path.join(path, 'train.csv'))

# %%
df.head()

# %%
n_imgs = len(df)
n_imgs

# %%
plt.hist(df['distance'], bins=30)
plt.show()

# %%
img_width = img_height = 512
batch_size = 32

# %%
image_paths = []
for f in os.listdir(os.path.join(path, 'train')):
    if f.startswith('.') or '.jpg' not in f:
        continue
    image_paths.append(os.path.join(path, 'train', f))
len(image_paths)

# %%
distances = df.get('distance').tolist()
locations = df.get('location').tolist()

distances[:5]

# %%
image_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

# %%
combined_distances = list(zip(image_paths, distances))
combined_locations = list(zip(image_paths, locations))
random.Random(42).shuffle(combined_distances)
random.Random(42).shuffle(combined_locations)

# %%
image_paths_distances, distances = zip(*combined_distances)
image_paths_locations, locations = zip(*combined_locations)

# %%
train_split = int(0.7*n_imgs)
val_split = train_split + int(0.1*n_imgs)
test_split = val_split + int(0.2*n_imgs)
print(train_split, val_split, test_split)

# %%
train_image_paths_distances = np.array(image_paths_distances[:train_split]).flatten()
train_distances = np.array(distances[:train_split]).flatten()
train_image_paths_locations = np.array(image_paths_locations[:train_split]).flatten()
train_locations = locations[:train_split]

val_image_paths_distances = np.array(image_paths_distances[train_split:val_split]).flatten()
val_distances = np.array(distances[train_split:val_split]).flatten()
val_image_paths_locations = np.array(image_paths_locations[train_split:val_split]).flatten()
val_locations = locations[train_split:val_split]

test_image_paths_distances = np.array(image_paths_distances[val_split:test_split]).flatten()
test_distances = np.array(distances[val_split:test_split]).flatten()
test_image_paths_locations = np.array(image_paths_locations[val_split:test_split]).flatten()
test_locations = locations[val_split:test_split]

# %%
train_locations = tuple(map(lambda x: tuple(json.loads(x)), train_locations))
train_locations = np.array(train_locations)
train_locations = (train_locations-np.min(train_locations))/(np.max(train_locations)-np.min(train_locations))

val_locations = tuple(map(lambda x: tuple(json.loads(x)), val_locations))
val_locations = np.array(val_locations)
val_locations = (val_locations-np.min(val_locations))/(np.max(val_locations)-np.min(val_locations))

test_locations = tuple(map(lambda x: tuple(json.loads(x)), test_locations))
test_locations = np.array(test_locations)
test_locations = (test_locations-np.min(test_locations))/(np.max(test_locations)-np.min(test_locations))

# %%
train_distances = (train_distances-np.min(train_distances))/(np.max(train_distances)-np.min(train_distances))
val_distances = (val_distances-np.min(val_distances))/(np.max(val_distances)-np.min(val_distances))
test_distances = (test_distances-np.min(test_distances))/(np.max(test_distances)-np.min(test_distances))

# %%
print(np.max(train_locations))
print(np.min(train_locations))

# %%
print(np.max(train_distances))
print(np.min(train_distances))

# %%
def as_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image

# %%
train_image_paths_distances = [as_image(_) for _ in train_image_paths_distances]
val_image_paths_distances = [as_image(_) for _ in val_image_paths_distances]
test_image_paths_distances = [as_image(_) for _ in test_image_paths_distances]

train_image_paths_distances = np.array(train_image_paths_distances)
val_image_paths_distances = np.array(val_image_paths_distances)
test_image_paths_distances = np.array(test_image_paths_distances)

# %%
print(train_image_paths_distances.shape)

# %%
net = tf.keras.applications.MobileNetV3Small(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
net.trainable = False

# %%
# If this is commented out, it is to reduce file length as this output is long
# net.summary()

# %%
lr = 1e-4
epochs = 25

flatten = net.output
flatten = tf.keras.layers.Flatten()(flatten)

regression = tf.keras.layers.Dense(64, activation='relu')(flatten)
regression = tf.keras.layers.Dense(32, activation='relu')(regression)
regression = tf.keras.layers.Dense(16, activation='relu')(regression)

# DISTANCE REGRESSION
left_head = tf.keras.layers.Dense(1, activation='relu', name='distance')(regression)

# LOCATION REGRESSION
right_head = tf.keras.layers.Dense(2, activation='relu', name='location')(regression)

model = tf.keras.Model(inputs=net.input, outputs=(left_head, right_head))

losses = { 'distance': 'mean_absolute_error', 'location': 'mean_squared_error' }
loss_weights = { 'distance': 1.0, 'location': 1.0 }

opt = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(loss=losses, optimizer=opt, metrics=['mae', 'mse'], loss_weights=loss_weights)

# If this is commented out, it is to reduce file length as this output is long
print(model.summary())

# %%
train_targets = { 'distance': train_distances, 'location': train_locations }
test_targets = { 'distance': test_distances, 'location': test_locations }
val_targets = { 'distance': val_distances, 'location': val_locations }

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
             tf.keras.callbacks.ModelCheckpoint(filepath='best_model_2.keras', monitor='val_loss', save_best_only=True)]
history = model.fit(train_image_paths_distances, train_targets,
                    validation_data = (val_image_paths_distances, val_targets),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks)

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss (MAE + MSE)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('loss2.png')
plt.show()


# %%
history.history.keys()

# %%
plt.plot(history.history['distance_mae'])
plt.plot(history.history['val_distance_mae'])
plt.title('Model Distance MAE')
plt.ylabel('Mean Absolute Error (MAE)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('mae2.png')
plt.show()

# %%
plt.plot(history.history['location_mse'])
plt.plot(history.history['val_location_mse'])
plt.title('Model Location MSE')
plt.ylabel('Location Mean Squared Error (MSE)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.savefig('mse2.png')
plt.show()

# %%
model.evaluate(test_image_paths_distances, test_targets)

# %%



