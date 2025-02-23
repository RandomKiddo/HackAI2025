import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

model_file_path = 'MODEL NAME GOES HERE'
csv_file_path = 'CSV FILE PATH GOES HERE'

# Load the trained model
model = tf.keras.models.load_model(
    'keypoint_model6.keras',
    custom_objects={
        "euclidean_distance": lambda y_true, y_pred: tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true[:, :2] - y_pred[:, :2]), axis=1))),
        "depth_difference": lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_true[:, 2] - y_pred[:, 2]))
    }
)

# Load CSV for ground truth labels
df = pd.read_csv(csv_file_path)

# Ensure (x, y) columns are parsed only once
df["x"] = df["location"].apply(lambda xy: int(xy.strip("[]").split(",")[0]))
df["y"] = df["location"].apply(lambda xy: int(xy.strip("[]").split(",")[1]))

# Image parameters
IMG_SIZE = 512
IMG_WIDTH, IMG_HEIGHT = 512, 512  # Original image size

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    resized_img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    normalized_img = resized_img / 255.0  # Normalize pixel values
    return img_rgb, np.expand_dims(normalized_img, axis=0)

# Function to get ground truth (x, y) from the CSV
def get_ground_truth(image_id):
    # Ensure image_id is integer
    image_id = int(image_id)
    
    # Retrieve the corresponding row by ImageID
    row = df[df["ImageID"] == image_id]
    if row.empty:
        raise ValueError(f"Image ID {image_id} not found in CSV.")

    # Extract the (x, y) coordinates
    return int(row["x"].values[0]), int(row["y"].values[0])

# Function to draw the keypoint (predicted and ground truth) on the image
def draw_keypoints(image, prediction, true_x, true_y):
    # Extract predicted (x, y)
    pred_x, pred_y = prediction[0]
    
    # Scale x, y back to original image size
    pred_x = int(pred_x * IMG_WIDTH)
    pred_y = int(pred_y * IMG_HEIGHT)

    # Draw a red "X" for the predicted keypoint
    line_length = 3
    cv2.line(image, (pred_x - line_length, pred_y - line_length), (pred_x + line_length, pred_y + line_length), (255, 165, 0), 2)
    cv2.line(image, (pred_x - line_length, pred_y + line_length), (pred_x + line_length, pred_y - line_length), (255, 165, 0), 2)

    # Draw a green circle for the ground truth keypoint
    cv2.circle(image, (true_x, true_y), 30, (0, 255, 0), 2)

    return image

# Main function to run the demonstration
for i in range(50):
    print("")
while True:
    IMAGE_FOLDER = "data/"
    IMAGE_NUM = input("Enter the picture number you would like to use (or 'exit' to quit): ")

    if IMAGE_NUM.lower() == "exit":
        break

    image_path = IMAGE_FOLDER + IMAGE_NUM + ".jpg"
    try:
        original_img, input_img = preprocess_image(image_path)

        # Get ground truth coordinates
        true_x, true_y = get_ground_truth(IMAGE_NUM)

        # Predict keypoint
        prediction = model.predict(input_img)

        # Draw keypoints on the image
        output_img = draw_keypoints(original_img, prediction, true_x, true_y)

        # Display the result
        plt.imshow(output_img)
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

print("Exiting...")
print("")
print(model.summary())