import numpy as np
import matplotlib.pyplot as plt

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

def get_adjacency_matrix(pose_points):
    # Define the number of nodes (pose estimation points)
    num_nodes = pose_points

    # Initialize an empty adjacency matrix with zeros
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Set entries in the adjacency matrix to 1 for each connection
    for connection in ALL_CONNECTIONS:
        node1, node2 = connection
        adjacency_matrix[node1, node2] = 1
        adjacency_matrix[node2, node1] = 1  # Since the graph is undirected, set both directions

    # Print the adjacency matrix (optional)
    print(adjacency_matrix)
    print(adjacency_matrix.shape)

    return adjacency_matrix

