import os
import random
import json
import psutil
import time

import tensorflow as tf
import kagglehub as kh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t0 = time.time()
print('Process Started')

def print_mem():
    process = psutil.Process(os.getpid())
    print(round(process.memory_info().rss/(10**9), 3), 'Gbytes')

print('Starting Memory:')
print_mem()

tf.random.set_seed(42)

path = kh.dataset_download("msafi04/iss-docking-dataset")

df = pd.read_csv(os.path.join(path, 'train.csv'))

print('DF Read:')
print_mem()

image_paths = []
for f in os.listdir(os.path.join(path, 'train')):
    if f.startswith('.') or '.jpg' not in f:
        continue
    image_paths.append(os.path.join(path, 'train', f))
len(image_paths)

distances = df.get('distance').tolist()
locations = df.get('location').tolist()

image_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

print('Images and Values Fetched:')
print_mem()

combined_distances = list(zip(image_paths, distances))
combined_locations = list(zip(image_paths, locations))
random.Random(42).shuffle(combined_distances)
random.Random(42).shuffle(combined_locations)

image_paths_distances, distances = zip(*combined_distances)
image_paths_locations, locations = zip(*combined_locations)

print('Values Shuffled and Unzipped:')
print_mem()

n_imgs = len(df)
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

print('Train/Test/Val Split Made:')
print_mem()

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

print('Data Normalized:')
print_mem()

def as_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def create_dataset(paths, dists, locs):
    dataset = tf.data.Dataset.from_tensor_slices((paths, dists, locs))
    dataset = dataset.map(lambda i, d, xy: (as_image(i), {'distance': d, 'x': xy[0], 'y': xy[1]}), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_image_paths_distances, train_distances, train_locations)

print('Train Read (Assummed Intensive):')
print_mem()

val_dataset = create_dataset(val_image_paths_distances, val_distances, val_locations)

print('Val Read (Assummed Intensive):')
print_mem()

test_dataset = create_dataset(test_image_paths_distances, test_distances, test_locations)

print('Test Read (Assummed Intensive):')
print_mem()

print('All Images Read (Assummed Intensive):')
print_mem()

time = time.time()
print('Time Difference (s):')
print(time-t0)