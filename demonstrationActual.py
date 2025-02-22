import trimesh
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# 1. Load the STL file
stl_file = "ISS.stl"  # Update with your actual file path
mesh = trimesh.load_mesh(stl_file)

# 2. Create a plotter using PyVista
plotter = pv.Plotter()
pv_mesh = pv.wrap(mesh)

# 3. Add STL geometry
plotter.add_mesh(pv_mesh, color="lightblue", opacity=1)

# 4. Overlay a 2D Line
x = [0, 10]
y = [2, 30]
z = [0, 0]  # Assuming a flat 2D line in 3D space

line = pv.Line((x[0], y[0], z[0]), (x[1], y[1], z[1]))
plotter.add_mesh(line, color="red", line_width=5)

# 5. Show the plot
plotter.show()