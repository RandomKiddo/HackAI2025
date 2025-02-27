import os
import random
import json
import pickle

import tensorflow as tf
import kagglehub as kh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

tf.random.set_seed(42)

path = kh.dataset_download("msafi04/iss-docking-dataset")

df = pd.read_csv(os.path.join(path, 'train.csv'))

n_imgs = len(df)

img_width = img_height = 512
batch_size = 32

image_paths = []
for f in os.listdir(os.path.join(path, 'train')):
    if f.startswith('.') or '.jpg' not in f:
        continue
    image_paths.append(os.path.join(path, 'train', f))
len(image_paths)

distances = df.get('distance').tolist()
locations = df.get('location').tolist()

image_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

combined_distances = list(zip(image_paths, distances))
combined_locations = list(zip(image_paths, locations))
random.Random(42).shuffle(combined_distances)
random.Random(42).shuffle(combined_locations)

image_paths_distances, distances = zip(*combined_distances)
image_paths_locations, locations = zip(*combined_locations)

train_split = int(0.7*n_imgs)
val_split = train_split + int(0.1*n_imgs)
test_split = val_split + int(0.2*n_imgs)

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

train_locations = tuple(map(lambda x: tuple(json.loads(x)), train_locations))
train_locations = np.array(train_locations)/512

val_locations = tuple(map(lambda x: tuple(json.loads(x)), val_locations))
val_locations = np.array(val_locations)/512

test_locations = tuple(map(lambda x: tuple(json.loads(x)), test_locations))
test_locations = np.array(test_locations)/512

max_distance = np.max(distances)
min_distance = np.min(distances)

train_distances = (train_distances-min_distance)/(max_distance-min_distance)
val_distances = (val_distances-min_distance)/(max_distance-min_distance)
test_distances = (test_distances-min_distance)/(max_distance-min_distance)

def as_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (img_height, img_width))
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
    return image

def create_dataset(paths, dists, locs):
    dataset = tf.data.Dataset.from_tensor_slices((paths, dists, locs))
    dataset = dataset.map(lambda i, d, xy: (as_image(i), {'distance': d, 'x': xy[0], 'y': xy[1]}), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_image_paths_distances, train_distances, train_locations)
val_dataset = create_dataset(val_image_paths_distances, val_distances, val_locations)
test_dataset = create_dataset(test_image_paths_distances, test_distances, test_locations)

class TestModel(tf.keras.Model):
    def __init__(self):
        super(TestModel, self).__init__()

        self.net = tf.keras.applications.MobileNetV3Small(
            input_shape=(img_height, img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        self.net.trainable = False

        self.flatten = tf.keras.layers.Flatten()
        self.regression1 = tf.keras.layers.Dense(32, activation='relu')
        self.regression2 = tf.keras.layers.Dense(16, activation='relu')

        self.left_head1 = tf.keras.layers.Dense(8, activation='relu')
        self.left_head2 = tf.keras.layers.Dense(1, activation='relu', name='distance')

        self.middle_head1 = tf.keras.layers.Dense(8, activation='relu')
        self.middle_head2 = tf.keras.layers.Dense(1, activation='relu', name='x')

        self.right_head = tf.keras.layers.Dense(1, activation='relu', name='y')
    
    def call(self, inputs):
        x = self.net(inputs)
        x = self.flatten(x)
        x = self.regression1(x)
        x = self.regression2(x)

        left_head = self.left_head1(x)
        left_head = self.left_head2(left_head)

        middle_head = self.middle_head1(x)
        middle_head = self.middle_head2(middle_head)

        right_head = self.right_head(x)

        return left_head, middle_head, right_head

model = TestModel()
loss_object = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(1e-10)

def lr_schedule(epoch):
    return 1e-10 * 10**(epoch/10)

@tf.function
def train_step(model, optimizer, inputs, targets, loss_object):
    with tf.GradientTape() as tape:
        l, m, r = model(inputs)
        loss = loss_object(targets['distance'], l)
        loss += loss_object(targets['x'], m)
        loss += loss_object(targets['y'], r)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

lrs, losses = [], []
for epoch in range(100):
    lr = lr_schedule(epoch)
    optimizer.learning_rate.assign(lr)

    total_loss = 0
    for i, j in zip(tqdm(range(7000//batch_size+1)), train_dataset):
        inputs = j[0]
        targets = j[1]

        loss = train_step(model, optimizer, inputs, targets, loss_object)
        total_loss += loss
    
    lrs.append(lr)
    losses.append(total_loss.numpy())

plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.savefig('lrfinder.py')
plt.show()