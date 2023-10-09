import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import os

# left_hand_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# right_hand_idx = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
# whole_body_idx = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]

HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))
HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))
HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))
HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))
HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))
HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

# Define additional connections for the right hand starting at index 21
RIGHT_HAND_PALM_CONNECTIONS = ((21, 22), (21, 26), (30, 34), (34, 38), (26, 30), (21, 38))
RIGHT_HAND_THUMB_CONNECTIONS = ((22, 23), (23, 24), (24, 25))
RIGHT_HAND_INDEX_FINGER_CONNECTIONS = ((26, 27), (27, 28), (28, 29))
RIGHT_HAND_MIDDLE_FINGER_CONNECTIONS = ((30, 31), (31, 32), (32, 33))
RIGHT_HAND_RING_FINGER_CONNECTIONS = ((34, 35), (35, 36), (36, 37))
RIGHT_HAND_PINKY_FINGER_CONNECTIONS = ((38, 39), (39, 40), (40, 41))

# Combine all hand connections into one set
HAND_CONNECTIONS = frozenset().union(*[
    HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS,
    HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS,
    HAND_RING_FINGER_CONNECTIONS, HAND_PINKY_FINGER_CONNECTIONS,
    RIGHT_HAND_PALM_CONNECTIONS, RIGHT_HAND_THUMB_CONNECTIONS,
    RIGHT_HAND_INDEX_FINGER_CONNECTIONS, RIGHT_HAND_MIDDLE_FINGER_CONNECTIONS,
    RIGHT_HAND_RING_FINGER_CONNECTIONS, RIGHT_HAND_PINKY_FINGER_CONNECTIONS
])

POSE_CONNECTIONS = frozenset([(42, 43), (43, 44), (44, 45), (45, 49), (42, 46), (46, 47),
                              (47, 48), (48, 50), (51, 52), (53, 54), (53, 55),
                              (55, 57), (54, 56), (56, 58)])

# Combine all pose and hand connections into one set
ALL_CONNECTIONS = frozenset().union(POSE_CONNECTIONS, HAND_CONNECTIONS)


def connect_points(img, x, y, connections, color):
    """
    Connect specified points in an image with lines based on the connection definitions.

    Args:
    img (numpy array): The image on which to draw the lines.
    x (numpy array): The x-coordinates of all points.
    y (numpy array): The y-coordinates of all points.
    connections (set): The set of connections to draw.
    color (tuple): The color of the lines in BGR format.

    Returns:
    None
    """
    for connection in connections:
        pt1 = (x[connection[0]], y[connection[0]])
        pt2 = (x[connection[1]], y[connection[1]])
        cv2.line(img, pt1, pt2, color, 1)


def plot_2d_pose_estimations(frames_data, class_name, image_size=(256, 256)):
    """
    Map 2D pose estimations onto a blank image for each frame, connect specified points, and display using Matplotlib.

    Args:
    frames_data (numpy array): An array with shape [frames, 1, 59, 3],
        where frames is the number of frames, and each frame contains 59 points
        with 3 coordinates (x, y, z).
    image_size (tuple): Size of the output image in the format (width, height).

    Returns:
    None
    """
    print(class_name)
    frames_data = frames_data.permute(1, 3, 2, 0)
    num_frames = frames_data.shape[0]
    num_points = frames_data.shape[2]
    for frame_idx in range(num_frames):
        # Create a blank image
        img_points = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        img_graph = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        # Extract x and y coordinates for each body point for the current frame
        frame_data = frames_data[frame_idx, 0, :, :]
        x = frame_data[:, 0]
        y = frame_data[:, 1]

        # Convert TensorFlow tensors to NumPy arrays and scale the coordinates
        x_scaled = (x.numpy() * image_size[0]).astype(int)
        y_scaled = (y.numpy() * image_size[1]).astype(int)

        # Draw the pose points on the image
        for i in range(num_points):
            cv2.circle(img_points, (x_scaled[i], y_scaled[i]), 2, (0, 255, 0), 1)  # Draw a green circle
            cv2.circle(img_graph, (x_scaled[i], y_scaled[i]), 2, (0, 255, 0), 1)

            # Connect specified points with lines (e.g., left hand, right hand, whole body)
            # Connect specified points based on the connection definitions
        connect_points(img_graph, x_scaled, y_scaled, ALL_CONNECTIONS, (255, 255, 255))  # Red lines

        # Display the image using Matplotlib
        plt.imshow(cv2.cvtColor(img_points, cv2.COLOR_BGR2RGB))
        plt.title(f"2D Pose Estimation - Frame {frame_idx+1}")
        plt.axis('off')  # Hide axis
        # plt.savefig(f"output_frames/{class_name}_frame_points_{frame_idx+1}.png")
        cv2.imwrite(f"output_frames/{class_name}_frame_points_{frame_idx+1}.png", img_points)

        plt.imshow(cv2.cvtColor(img_graph, cv2.COLOR_BGR2RGB))
        plt.title(f"2D Pose Estimation - Graph {frame_idx + 1}")
        plt.axis('off')  # Hide axis
        plt.savefig(f"output_frames/{class_name}_frame_graph_{frame_idx + 1}.png")
        cv2.imwrite(f"output_frames/{class_name}_frame_graph_{frame_idx + 1}.png", img_graph)

        # plt.show()
