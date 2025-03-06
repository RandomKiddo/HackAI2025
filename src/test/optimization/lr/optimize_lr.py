"""
TEST FILE

THIS FILE CURRENTLY RUNS TF.FUNCTION CODE WITH GRADIENT TAPE TO FIND THE
OPTIMIZED LEARNING RATE FOR THE HACKAI MODEL OF THE MHN. IT OUTPUTS A 
PNG PLOT SHOWING LOSS VS. LEARNING RATE TO EXAMINE FOR OPTIMAL LEARNING
RATE.

THIS FILE IS OPTIMIZED FOR USE WITH A GPU WITH MEMORY GROWTH, UTILIZING
TENSORFLOW MIXED PRECISION POLICY, AND NOT USING THE CACHED DATASET TO
SAVE RAM. 
"""

# MODE DEFINES WHICH MODEL
# 1 = HACKAI MODEL
# 2 = HEATMAP MODEL
# 3 = DISTANCE HEAD ONLY MODEL (v1)
# 4 = DISTANCE HEAD ONLY MODEL (v2)
MODE = 3

import os
import random
import json
import gc

import tensorflow as tf
import kagglehub as kh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from models import *
from functions import *

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)

tf.random.set_seed(42)

path = kh.dataset_download("msafi04/iss-docking-dataset")

df = pd.read_csv(os.path.join(path, 'train.csv'))

n_imgs = len(df)

img_width = img_height = 512
batch_size = 64

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
    #dataset = dataset.cache() # COMMENTED FOR RAM OPTIMIZATION
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

optimizer = tf.keras.optimizers.Adam(1e-10)

if MODE == 1:
    model = HackAIModel()
    loss_object_dist = tf.keras.losses.MeanAbsoluteError()
    loss_object_x = tf.keras.losses.MeanAbsoluteError()
    loss_object_y = tf.keras.losses.MeanAbsoluteError()

    train_dataset = create_dataset(train_image_paths_distances, train_distances, train_locations)
    val_dataset = create_dataset(val_image_paths_distances, val_distances, val_locations)
    test_dataset = create_dataset(test_image_paths_distances, test_distances, test_locations)
elif MODE == 2:
    model = HeatmapModel()
    loss_object_dist = tf.keras.losses.MeanAbsoluteError()
    loss_object_x = tf.keras.losses.CategoricalCrossentropy()
    loss_object_y = tf.keras.losses.CategoricalCrossentropy()

    train_x_heatmaps, train_y_heatmaps = convert_labels_to_heatmaps(train_locations[:, 0] * 512, train_locations[:, 1] * 512)
    val_x_heatmaps, val_y_heatmaps = convert_labels_to_heatmaps(val_locations[:, 0] * 512, val_locations[:, 1] * 512)
    test_x_heatmaps, test_y_heatmaps = convert_labels_to_heatmaps(test_locations[:, 0] * 512, test_locations[:, 1] * 512)

    train_dataset = create_dataset(train_image_paths_distances, train_distances, (train_x_heatmaps, train_y_heatmaps))
    val_dataset = create_dataset(val_image_paths_distances, val_distances, (val_x_heatmaps, val_y_heatmaps))
    test_dataset = create_dataset(test_image_paths_distances, test_distances, (test_x_heatmaps, test_y_heatmaps))
elif MODE == 3 or MODE == 4:
    if MODE == 3:
        model = DistanceHeadOnlyModelv1()
    else:
        model = DistanceHeadOnlyModelv2()
    loss_object_dist = tf.keras.losses.MeanSquaredError()
    loss_object_x = loss_object_y = None 

    train_x_heatmaps, train_y_heatmaps = convert_labels_to_heatmaps(train_locations[:, 0] * 512, train_locations[:, 1] * 512)
    val_x_heatmaps, val_y_heatmaps = convert_labels_to_heatmaps(val_locations[:, 0] * 512, val_locations[:, 1] * 512)
    test_x_heatmaps, test_y_heatmaps = convert_labels_to_heatmaps(test_locations[:, 0] * 512, test_locations[:, 1] * 512)

    train_dataset = create_dataset(train_image_paths_distances, train_distances, (train_x_heatmaps, train_y_heatmaps))
    val_dataset = create_dataset(val_image_paths_distances, val_distances, (val_x_heatmaps, val_y_heatmaps))
    test_dataset = create_dataset(test_image_paths_distances, test_distances, (test_x_heatmaps, test_y_heatmaps))
else:
    raise ValueError('MODE value must be integer in range [1, 4].')

@tf.function
def train_step(model, optimizer, inputs, targets, loss_obj_dist, loss_obj_x=None, loss_obj_y=None):
    with tf.GradientTape() as tape:
        tape.watch(inputs)

        l, m, r = model(inputs)
        loss = loss_obj_dist(targets['distance'], l)
        if loss_obj_x is not None:
            loss += loss_obj_x(targets['x'], m)
        if loss_obj_y is not None:
            loss += loss_obj_y(targets['y'], r)

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

        loss = train_step(model, optimizer, inputs, targets, loss_object_dist, loss_object_x, loss_object_y)
        total_loss += loss

        del inputs, targets, loss
        gc.collect()
    
    lrs.append(lr)
    losses.append(total_loss.numpy())

plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.savefig(f'lrfinder{MODE}.png')
plt.show()