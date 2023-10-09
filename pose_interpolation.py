import numpy as np
import pandas as pd


def interpolate_pose_frames(pose_npy, body):
    # convert to pandas dataframe
    frames = pose_npy.shape[0]
    points = pose_npy.shape[1]
    pose_npy = pose_npy.reshape((pose_npy.shape[0], pose_npy.shape[1] * pose_npy.shape[2]))
    # print(pose_npy.shape)
    pose_df = pd.DataFrame(pose_npy)
    # print(pose_df.head(10))
    # interpolate missing values
    pose_df.interpolate(method='index', axis=0, inplace=True)
    pose_df.bfill(axis='index', inplace=True)

    pose = pose_df.to_numpy()
    # print(np.count_nonzero(np.isnan(pose)))
    # print(pose.shape)
    # print(pose)
    if np.isnan(pose).any():
        print("NaN values still present in " + body)
    else:
        return pose.reshape((frames, points, 3))
