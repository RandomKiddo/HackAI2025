import trimesh
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import kagglehub as kh
import pandas as pd

# Function to create a rotation matrix
def get_rotation_matrix(axis, degrees):
    radians = np.radians(degrees)
    return trimesh.transformations.rotation_matrix(radians, axis)

# Get path variable from kaggle
path = kh.dataset_download("msafi04/iss-docking-dataset")
# print(path)

# Define dataset paths
IMAGE_FOLDER = path + "/train"
CSV_FILE = path + "/train.csv"

# Define stl paths
iss_stl_file = "ISS.stl"
dragon_stl_file = "dragon.stl"

df = pd.read_csv(CSV_FILE)
# Ensure (x, y) columns are parsed only once
df["x"] = df["location"].apply(lambda xy: int(xy.strip("[]").split(",")[0]))
df["y"] = df["location"].apply(lambda xy: int(xy.strip("[]").split(",")[1]))
# Remove old location column
del df["location"]
print(df)

IMAGE_NUM = input("Enter the picture number you would like to use: ")
IMAGE_NUM = int(IMAGE_NUM)
groundTruth = df[df["ImageID"] == IMAGE_NUM]
groundTruth = [int(groundTruth["x"].values[0]), int(groundTruth["y"].values[0]), int(groundTruth["distance"].values[0])]
print(groundTruth)

groundTruth[0] = groundTruth[0]/32
groundTruth[1] = groundTruth[1]/32
groundTruth[2] = groundTruth[2]/2.75

# prediction = model.predict(input_img) # Variable 'prediction' is a list in the format of [Z, X, Y]
prediction = groundTruth

# Create initial offsets, so that the docking port's of each spacecraft are centered at the origin
issOffset = [-5.25, 1.19, -0.105]
dragonOffset = [3.15, 0, 0]
issCoordinates = [0 + issOffset[0], 0+ issOffset[1], 0 + issOffset[2]]
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

'''
# Create a plotter using PyVista
plotter = pv.Plotter()
iss_pv_mesh = pv.wrap(issMesh)
dragon_pv_mesh = pv.wrap(dragonMesh)

# Overlay a 3D arrow, based off of our true value
line = pv.Line(pointa = [0,-8,-8], pointb = [-1*prediction[2], -1*prediction[1], -1*prediction[0]])

# Add STL geometry
plotter.add_mesh(iss_pv_mesh, color="lightblue", opacity=1)
plotter.add_mesh(dragon_pv_mesh, color="white", opacity=0.9)
plotter.add_mesh(line, color="lightgreen", opacity=1, line_width=3)

# We used this code snippet to help determing our offsets
# sphere = pv.Sphere(radius=0.25,center=(0,0,0))
# plotter.add_mesh(sphere, color = "red")

# Show the plot
plotter.show_axes()
plotter.add_background_image('background.jpg')
plotter.show()
'''

# Create a plotter using PyVista
plotter = pv.Plotter()
iss_pv_mesh = pv.wrap(issMesh)
dragon_pv_mesh = pv.wrap(dragonMesh)

# Overlay a 3D arrow, based off of our true value
line = pv.Line(pointa = [0,-8,-8], pointb = [-1*prediction[2], -1*prediction[1], -1*prediction[0]])

# Add STL geometry
plotter.add_mesh(iss_pv_mesh, color="lightblue", opacity=1)
plotter.add_mesh(dragon_pv_mesh, color="white", opacity=0.9)
plotter.add_mesh(line, color="lightgreen", opacity=1, line_width=3)
plotter.add_mesh(iss_pv_mesh, color="lightblue", opacity=1)
plotter.add_mesh(dragon_pv_mesh, color="white", opacity=0.9)
plotter.add_mesh(line, color="lightgreen", opacity=1, line_width=3)
plotter.show_axes()
plotter.add_background_image('background.jpg')

# Set Initial Camera Position at the Tip of Dragon Shuttle
initial_camera_position = [dragonCoordinates[0] - 3.15, dragonCoordinates[1], dragonCoordinates[2]]  # Slightly ahead of the nose
focal_point = issCoordinates  # Look at the ISS
view_up = [0, 1, 0]  # Keep the camera upright

plotter.camera_position = [initial_camera_position, focal_point, view_up]
plotter.show(auto_close=False)  # Keep window open for animation

# Animate Pullback to Isometric View
num_frames = 60  # Number of animation frames
while True:
    for i in range(num_frames):
        # Interpolate between initial and final camera position
        alpha = i
        new_camera_position = [
            initial_camera_position[0] + alpha * 10,  # Move back in X
            initial_camera_position[1] - alpha * 10,  # Move back in Y
            initial_camera_position[2] + alpha * 5,   # Move up in Z
        ]
        
        plotter.camera_position = [new_camera_position, focal_point, view_up]
        plotter.update()
        plotter.render()  # Ensure updates appear smoothly

plotter.close()  # Close the window after animation