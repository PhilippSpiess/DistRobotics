# Copyright (c) 2024 Philipp Spiess
# All rights reserved.

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

from helpers import timeit
from config_simulation import *

PLOT = True  # Set to True to plot the images with detections
VISION_MODEL = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
DEPTH_MODEL = "https://tfhub.dev/intel/midas/v2/2"

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.set_logical_device_configuration(
        gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
    )
        
if not os.path.exists(VISION_PATH+"/detector"):
    # Download and save the model locally
    model = hub.load(VISION_MODEL)
    tf.saved_model.save(model, VISION_PATH+"/detector")
    print(f"Model downloaded and saved to {VISION_PATH}")
else:
    print(f"Model already exists at {VISION_PATH}")

# Load the object detection model locally
VISION_MODEL = hub.load(VISION_PATH+"/detector_resnet")
detector = VISION_MODEL.signatures['default']

if not os.path.exists(VISION_PATH+"/depth"):
    # Download and save the model locally
    model = hub.load(DEPTH_MODEL, tags=['serve']).signatures['serving_default']
    tf.saved_model.save(model, VISION_PATH+"/depth")
    print(f"Model downloaded and saved to {VISION_PATH}")
else:
    print(f"Model already exists at {VISION_PATH}")

DEPTH_MODEL = tf.saved_model.load(VISION_PATH+"/depth")
depth_detector = DEPTH_MODEL

# Load the category mapping file
CLASS_MAPPING = {}
with open(VISION_PATH+'/class-descriptions-boxable.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        CLASS_MAPPING[row[0]] = row[1]

def preprocess_image(data):
    img_resized = tf.image.resize(data, [384, 384])
    img_expanded = tf.expand_dims(img_resized, axis=0)
    return img_expanded

def preprocess_depth_image(data):
    img_resized = tf.image.resize(data, [384, 384])
    img_expanded = tf.expand_dims(img_resized, axis=0)
    img_reordered = tf.transpose(img_expanded, [0, 3, 1, 2])  # Reorder to [1, 3, 384, 384]
    return img_reordered

# Optimized detection function
@timeit
@tf.function
def detect_object_(data):
    input_tensor = preprocess_image(data)
    result = detector(input_tensor)
    return result

@tf.function
def estimate_depth_(data):
    depth_input = preprocess_depth_image(data)
    depth_result = depth_detector(depth_input)['default']
    #depth_resized = tf.image.resize(depth_result, [data.shape[1], data.shape[2]])  # Resize to match original image
    return depth_result
    
def calculate_average_depth(depth_map, box, height, width):
    ymin, xmin, ymax, xmax = box
    ymin = int(ymin * height)
    xmin = int(xmin * width)
    ymax = int(ymax * height)
    xmax = int(xmax * width)

    # Extract the region of the depth map within the bounding box
    box_depth_map = np.array(depth_map)[0][ymin:ymax, xmin:xmax]

    # Calculate the average depth
    avg_depth = tf.reduce_mean(box_depth_map)
    return avg_depth.numpy()

def detect_object(data):

    try:
        height = 224
        width = 224
        data_rgb = data.reshape((height, width, 4))
    except:
        height = 1333
        width = 2000
        data_rgb = data.reshape((height, width, 3))
    data = data_rgb[:, :, :3]

    # Normalize the image data
    img_normalized = data.astype(np.float32) / 255.0

    result = detect_object_(img_normalized)
    depth_map = estimate_depth_(img_normalized)

    boxes = result['detection_boxes'].numpy()[:15]
    classes = result['detection_class_names'].numpy()[:15]
    classes = [CLASS_MAPPING.get(c.decode("utf-8"), "Unknown") for c in classes]
    scores = result['detection_scores'].numpy()[:15]

    # Convert data_rgb to uint8
    data_rgb = (data_rgb * 255).astype(np.uint8)
    
    average_depths = []
    for box in boxes:
        avg_depth = calculate_average_depth(depth_map, box, height, width)
        average_depths.append(float(avg_depth))

    if PLOT:
        # Plot depth map
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(depth_map.numpy()[0, :, :], cmap='inferno')
        plt.axis('off')
        plt.title("Depth Map")

        # Draw bounding boxes and labels on the image
        for i, (box, class_name, score) in enumerate(zip(boxes, classes, scores)):
            if i >= 3: break

            ymin, xmin, ymax, xmax = box
            ymin = int(ymin * height)
            xmin = int(xmin * width)
            ymax = int(ymax * height)
            xmax = int(xmax * width)

            cv2.rectangle(data_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{class_name}: {score:.2f}"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(ymin, label_size[1])
            cv2.rectangle(data_rgb, (xmin, top - label_size[1]), (xmin + label_size[0], top + base_line), (255, 255, 255), cv2.FILLED)
            cv2.putText(data_rgb, label, (xmin, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(data_rgb, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Object Detection")
        # Save the plot as an image file
        plt.savefig(VISION_PATH + "/detection_and_depth_map.png")

        # Optionally, you can close the figure to free up memory
        plt.close()

    data_save = {
        "coordinates": np.array(boxes).tolist(),
        "labels": classes,
        "values": np.array(scores).tolist(),
        "depths": average_depths  # Add the average depths here
    }

    with open(VISION_PATH + "/result.json", 'w') as f:
        json.dump(data_save, f)

    return scores

if __name__ == "__main__":

    while True:
        FPV = np.load(VISION_PATH+"/FPV.npy")
        detect_object(FPV)
        print("Object and depth detection completed.")

