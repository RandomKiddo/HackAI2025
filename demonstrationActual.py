import trimesh
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import kagglehub as kh
import pandas as pd
import tensorflow as tf
import os

# Function to create a rotation matrix
def get_rotation_matrix(axis, degrees):
    radians = np.radians(degrees)
    return trimesh.transformations.rotation_matrix(radians, axis)

# Load and preprocess a single image for prediction
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (512, 512))  # Ensure size matches model input
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Get path variable from kaggle
path = kh.dataset_download("msafi04/iss-docking-dataset")
# print(path)

model = tf.keras.models.load_model(
    'best_model_4.keras'
)

# Define dataset paths
IMAGE_FOLDER = path + "/train/"
CSV_FILE = path + "/train.csv"

# Define stl paths
iss_stl_file = "ISS.stl"
dragon_stl_file = "dragon.stl"

IMAGE_NUM = input("Enter the picture number you would like to use: ")
IMAGE_NUM = int(IMAGE_NUM)

# Provide an image path
image_path = os.path.join(IMAGE_FOLDER, f"{IMAGE_NUM}.jpg")
image = preprocess_image(image_path)  # Load and preprocess image

# Predict
predictions = np.array(model.predict(image)).flatten()

# Extract individual outputs
predicted_distance = predictions[0]  # Since model outputs a batch, take first value
predicted_x = predictions[1]
predicted_y = predictions[2]

# Scale x and y back to original range (if normalization was applied)
predicted_x *= 512
predicted_y *= 512
predicted_distance = predicted_distance*(491-61)+61

predictions[0] = predicted_distance
predictions[1] = predicted_x
predictions[2] = predicted_y
print(predictions)

# print(predicted_distance, predicted_x, predicted_y)

# Read train labels for ground truth positioning of the ISS
df = pd.read_csv(CSV_FILE)
# Ensure (x, y) columns are parsed only once
df["x"] = df["location"].apply(lambda xy: int(xy.strip("[]").split(",")[0]))
df["y"] = df["location"].apply(lambda xy: int(xy.strip("[]").split(",")[1]))
# Remove old location column
del df["location"]
print(df)

groundTruth = df[df["ImageID"] == IMAGE_NUM]
groundTruth = [int(groundTruth["x"].values[0]), int(groundTruth["y"].values[0]), int(groundTruth["distance"].values[0])]
# print(groundTruth)

groundTruth[0] = groundTruth[0]/32
groundTruth[1] = groundTruth[1]/32
groundTruth[2] = groundTruth[2]/2.75
print(groundTruth)

predictions[0] = predictions[0]/2.75
predictions[1] = predictions[1]/32
predictions[2] = predictions[2]/32
print(predictions)

# Create initial offsets, so that the docking port's of each spacecraft are centered at the origin
issOffset = [-5.25, 1.19, -0.105]
dragonOffset = [3.15, 0, 0]
issCoordinates = [0 + issOffset[0], 0 + issOffset[1], 0 + issOffset[2]]
dragonCoordinates = [dragonOffset[0], -8 + dragonOffset[1], -8 + dragonOffset[2]]

# Move the ISS to the Ground Truth location based off of our csv file data
issCoordinates = [issCoordinates[0] + -1*groundTruth[2], issCoordinates[1] + -1*groundTruth[1], issCoordinates[2] + -1*groundTruth[0]]

# Create initial Mesh's for both spacecraft
issMesh = trimesh.load_mesh(iss_stl_file)
issMesh.apply_translation(issCoordinates)
dragonMesh = trimesh.load_mesh(dragon_stl_file)
dragonMesh.apply_scale(0.004)
rotationMatrix = get_rotation_matrix([0, 0, 1], 90)
dragonMesh.apply_transform(rotationMatrix)
dragonMesh.apply_translation(dragonCoordinates)

# Create a plotter using PyVista
plotter = pv.Plotter(window_size=[512, 512])
iss_pv_mesh = pv.wrap(issMesh)
dragon_pv_mesh = pv.wrap(dragonMesh)

# Overlay a 3D arrow, based off of our true value
line = pv.Line(
    pointa = [0,-8,-8], 
    pointb = [
        -1*predictions[0], 
        -1*predictions[2], 
        -1*predictions[1]
    ]
)

# Add STL geometry
plotter.add_mesh(iss_pv_mesh, color="lightblue", opacity=1)
plotter.add_mesh(dragon_pv_mesh, color="white", opacity=0.9)
plotter.add_mesh(line, color="lightgreen", opacity=1, line_width=3)
plotter.show_axes()
# plotter.show()
# plotter.add_background_image('background.jpg')

plotter.open_gif("test.gif")

# Set Initial Camera Position at the Tip of Dragon Shuttle
initial_camera_position = [dragonCoordinates[0] - 3.15, dragonCoordinates[1], dragonCoordinates[2]]  # Slightly ahead of the nose
focal_point = issCoordinates  # Look at the ISS
view_up = [0, 1, 0]  # Keep the camera upright

plotter.camera_position = [initial_camera_position, focal_point, view_up]
# plotter.show(auto_close=False)  # Keep window open for animation

for i in range(10):
    plotter.write_frame()

for i in range(24):
    # Linear interpolation between initial and start of path
    if i<5:
        initial_camera_position[0] += 1
    elif i<10:
        initial_camera_position[0] += 2
    elif i<15:
        initial_camera_position[0] += 4
    else:
        initial_camera_position[0] += 8

    plotter.camera_position = [initial_camera_position, focal_point, view_up]
    plotter.write_frame()

# Create camera path
camera_path1 = pv.Line(
    pointa = initial_camera_position,
    pointb = [issCoordinates[0]-60, 20, -36],
    resolution = 20
)    

print(camera_path1)

# TO - DO: Convert orbit_on_path to a for loop that just goes along the points on a line using linspace along each axis
# THEN, IF possible, create the rotation and docking animation

plotter.orbit_on_path(
    camera_path1, 
    write_frames = True, 
    viewup = [0, 1, 0], 
    # focus = issCoordinates, 
    step=0.001
)