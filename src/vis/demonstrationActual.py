import trimesh
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import kagglehub as kh
import pandas as pd
import tensorflow as tf
import os
import cv2

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

def draw_keypoints(image, pred_x, pred_y, true_x, true_y):
    # Make sure that all values are ints
    pred_x, pred_y = int(pred_x), int(pred_y)
    true_x, true_y = int(true_x), int(true_y)

    # Draw a red "X" for the predicted keypoint
    line_length = 4
    cv2.line(image, (pred_x - line_length, pred_y - line_length), (pred_x + line_length, pred_y + line_length), (255, 255, 0), 2)
    cv2.line(image, (pred_x - line_length, pred_y + line_length), (pred_x + line_length, pred_y - line_length), (255, 255, 0), 2)

    # Draw a light green circle for the ground truth keypoint
    cv2.circle(image, (true_x, true_y), 30, (144, 238, 144), 2)

    return image

# Get path variable from kaggle
path = kh.dataset_download("msafi04/iss-docking-dataset")
# print(path)

model = tf.keras.models.load_model(
    'src/model.keras'
)

# Define dataset paths
IMAGE_FOLDER = path + "/train/"
CSV_FILE = path + "/train.csv"

# Define stl paths
iss_stl_file = "stl/ISS.stl"
dragon_stl_file = "stl/dragon.stl"

# Loop for convinience
while True:

    # Clear the screen
    for i in range(20): print("")

    IMAGE_NUM = input("Enter the picture number you would like to use: ")
    if IMAGE_NUM == "exit":
        print("Exiting...")
        break
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
    print("predictions: " + str(predictions) + "(z, x, y)")

    # print(predicted_distance, predicted_x, predicted_y)

    # Read train labels for ground truth positioning of the ISS
    df = pd.read_csv(CSV_FILE)
    # Ensure (x, y) columns are parsed only once
    df["x"] = df["location"].apply(lambda xy: int(xy.strip("[]").split(",")[0]))
    df["y"] = df["location"].apply(lambda xy: int(xy.strip("[]").split(",")[1]))
    # Remove old location column
    del df["location"]
    # print(df)

    groundTruth = df[df["ImageID"] == IMAGE_NUM]
    groundTruth = [int(groundTruth["x"].values[0]), int(groundTruth["y"].values[0]), int(groundTruth["distance"].values[0])]
    print("Ground Truth: " + str(groundTruth) + "(x, y, z)")

    groundTruth[0] = groundTruth[0]/32
    groundTruth[1] = groundTruth[1]/32
    groundTruth[2] = groundTruth[2]/2.75
    # print(groundTruth)

    predictions[0] = predictions[0]/2.75
    predictions[1] = predictions[1]/32
    predictions[2] = predictions[2]/32
    # print(predictions[1],predictions[2],predictions[0])

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
    dragonActor = plotter.add_mesh(dragon_pv_mesh, color="white", opacity=0.9)
    plotter.add_mesh(line, color="lightgreen", opacity=1, line_width=2)
    plotter.show_axes()
    # plotter.show()
    plotter.add_background_image('imgs/background.jpg')

    plotter.open_gif("test.gif")

    # Set Initial Camera Position at the Tip of Dragon Shuttle
    initial_camera_position = [dragonCoordinates[0] - 3.15, dragonCoordinates[1], dragonCoordinates[2]]  # Slightly ahead of the nose
    focal_point = issCoordinates  # Look at the ISS
    view_up = [0, 1, 0]  # Keep the camera upright

    plotter.camera_position = [initial_camera_position, focal_point, view_up]
    # plotter.show(auto_close=False)  # Keep window open for animation

    numFrames = 10
    for i in range(numFrames):
        plotter.write_frame()

    numFrames = 24
    for i in range(numFrames):
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

    focal_point = [dragonCoordinates[0] - 3.15, dragonCoordinates[1], dragonCoordinates[2]]  # Look at the tip of the dragon shuttle
    numFrames = 30
    # Reference value
    ref = issCoordinates[0]/2
    # Vector for camera path in the x direction
    cameraPathx = np.linspace(initial_camera_position[0], ref, numFrames, endpoint = True)
    # Vector for camera path in the x direction
    cameraPathy = np.linspace(initial_camera_position[1], 20, numFrames, endpoint = True)
    # Vector for camera path in the x direction
    cameraPathz = np.linspace(initial_camera_position[2], ref, numFrames, endpoint = True)

    for i in range(numFrames):
        initial_camera_position = [cameraPathx[i], cameraPathy[i], cameraPathz[i]]

        plotter.camera_position = [initial_camera_position, focal_point, view_up]
        plotter.write_frame()

    numFrames = 10
    for i in range(numFrames):
        plotter.write_frame()

    numFrames = 30
    # Vector for camera path in the x direction
    cameraPathx = np.linspace(initial_camera_position[0], -1*predictions[0], numFrames, endpoint = False)
    # Vector for camera path in the x direction
    cameraPathy = np.linspace(initial_camera_position[1], -1*predictions[2], numFrames, endpoint = False)
    # Vector for camera path in the x direction
    cameraPathz = np.linspace(initial_camera_position[2], -1*predictions[1], numFrames, endpoint = False)
    focal_point = [-1*predictions[0], -1*predictions[2], -1*predictions[1]] 
    for i in range(numFrames):
        initial_camera_position = [cameraPathx[i], cameraPathy[i], cameraPathz[i]]
        plotter.camera_position = [initial_camera_position, focal_point, view_up]
        plotter.write_frame()

    numFrames = 10
    for i in range(numFrames):
        plotter.write_frame()

    plotter.close()

    image_path = IMAGE_FOLDER + str(IMAGE_NUM) + ".jpg"
    original_img = cv2.imread(image_path)  # Read image as a NumPy array
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    output_img = draw_keypoints(
        original_img, 
        predictions[1]*32, 
        predictions[2]*32, 
        groundTruth[0]*32, 
        groundTruth[1]*32
    )
    plt.imsave("test.jpg", output_img, format="jpg")
    # Display the result
    plt.imshow(output_img)
    plt.axis("off")
    plt.show()