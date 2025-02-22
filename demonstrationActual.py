import trimesh
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# Function to create a rotation matrix
def get_rotation_matrix(axis, degrees):
    radians = np.radians(degrees)
    return trimesh.transformations.rotation_matrix(radians, axis)

csv_file_path = "iss.csv"

df = pd.read_csv(csv_file_path)

# Ensure (x, y) columns are parsed only once
df["x"] = df["location"].apply(lambda xy: int(xy.strip("[]").split(",")[0]))
df["y"] = df["location"].apply(lambda xy: int(xy.strip("[]").split(",")[1]))

# Create initial offsets, so that the docking port's of each spacecraft are centered at the origin
issOffset = [-5.25, 1.19, -0.105]
dragonOffset = [3.15, 0, 0]
issCoordinates = [0 + issOffset[0], 0+ issOffset[1], 0 + issOffset[2]]
dragonCoordinates = [0 + dragonOffset[0], 0 + dragonOffset[1], 0 + dragonOffset[2]]

# Load the STL file
iss_stl_file = "ISS.stl"
dragon_stl_file = "dragon.stl"

# Create initial Mesh's for both spacecraft
issMesh = trimesh.load_mesh(iss_stl_file)
issMesh.apply_translation(issCoordinates)
dragonMesh = trimesh.load_mesh(dragon_stl_file)
dragonMesh.apply_scale(0.004)
rotationMatrix = get_rotation_matrix([0, 0, 1], 90)
dragonMesh.apply_transform(rotationMatrix)
dragonMesh.apply_translation(dragonCoordinates)

# Create a plotter using PyVista
plotter = pv.Plotter()
iss_pv_mesh = pv.wrap(issMesh)
dragon_pv_mesh = pv.wrap(dragonMesh)

# Add STL geometry
plotter.add_mesh(iss_pv_mesh, color="lightblue", opacity=1)
plotter.add_mesh(dragon_pv_mesh, color="blue", opacity=1)

# We used this code snippet to help determing our offsets
# sphere = pv.Sphere(radius=0.25,center=(0,0,0))
# plotter.add_mesh(sphere, color = "red")

# Overlay a 3D arrow, based off of our true value
issOffset[i] = issOffset[i] + model.predict()[i]

# Show the plot
plotter.show_axes()
plotter.show()