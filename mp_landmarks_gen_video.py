import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import pandas as pd
from pose_interpolation import interpolate_pose_frames

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

pose_points = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE',
               'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
               'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_HIP', 'RIGHT_HIP']
print(len(pose_points))
# for landmark in mp_holistic.PoseLandmark:
#     print(landmark)
# print(len(pose_points))

# hand_points = []
hands = ['LEFT_HAND', 'RIGHT_HAND']
left_hand_points = []
right_hand_points = []
for hand in hands:
    for landmark in mp_holistic.HandLandmark:
        landmark = str(landmark)
        landmark = landmark.split(".")
        #     print(landmark[1])
        #     print(type(hand))
        point = hand + "_" + landmark[1]
        if hand == 'LEFT_HAND':
            left_hand_points.append(point)
        else:
            right_hand_points.append(point)

# print(left_hand_points)
# print(right_hand_points)

all_points = left_hand_points + right_hand_points + pose_points
# print(len(all_points))
# print(all_points)

# xyz points
all_points_xyz = []
co_ord = ['X', 'Y', 'Z']
for point in all_points:
    for i in range(3):
        new_point = ''
        new_point = point + "_" + co_ord[i]
        all_points_xyz.append(new_point)


# print(all_points_xyz)
# print(len(all_points_xyz))

def return_all_landmark_names():
    # return ['FRAME_NAME'] + all_points_xyz + ['CLASS']
    return ['FRAME_NAME'] + all_points + ['CLASS']


def generate_video_landmarks(video_path):
    video_name = os.path.split(video_path)[1]
    video_name = video_name.split(".")[0]
    #     print(video_name)
    all_video_landmarks = []
    left_hand = []
    right_hand = []
    pose = []

    cap = cv2.VideoCapture(str(video_path))
    frames = 0
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                # print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            pose_landmarks = results.pose_landmarks
            left_hand_landmarks = results.left_hand_landmarks
            right_hand_landmarks = results.right_hand_landmarks

            # Draw landmark annotation on the image.

            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # blank_image = np.zeros(image.shape)
            #
            # mp_drawing.draw_landmarks(
            #     blank_image,
            #     results.pose_landmarks,
            #     mp_holistic.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles
            #     .get_default_pose_landmarks_style())
            # mp_drawing.draw_landmarks(
            #     blank_image,
            #     results.right_hand_landmarks,
            #     mp_holistic.HAND_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles
            #     .get_default_pose_landmarks_style())
            # mp_drawing.draw_landmarks(
            #     blank_image,
            #     results.left_hand_landmarks,
            #     mp_holistic.HAND_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles
            #     .get_default_pose_landmarks_style())
            # # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Holistic', cv2.flip(blank_image, 1))
            # if cv2.waitKey(5) & 0xFF == 27:
            #   break

            # POSE landmarks
            if results.pose_landmarks:
                pose_data = {}
                for i in range(len(pose_points)):
                    results.pose_landmarks.landmark[i].x = results.pose_landmarks.landmark[i].x  # * image_width
                    results.pose_landmarks.landmark[i].y = results.pose_landmarks.landmark[
                        i].y  # *image_height using normalized values
                    results.pose_landmarks.landmark[i].z = results.pose_landmarks.landmark[i].z
                    pose_data.update(
                        {pose_points[i]: results.pose_landmarks.landmark[i]}
                    )

                keys = []
                for key in pose_data.keys():
                    keys.append(key)

                pose_xyz_data = np.empty((17, 3))

                index = 0
                for value in pose_data.values():
                    x = value.x
                    y = value.y
                    z = value.z

                    # append to numpy array
                    pose_xyz_data[index] = [x, y, z]

                    index += 1

            else:
                pose_xyz_data = np.empty((17, 3))
                pose_xyz_data[:] = np.nan

            # left hand landmarks
            if results.left_hand_landmarks:
                left_hand_data = {}
                for i in range(len(left_hand_points)):
                    results.left_hand_landmarks.landmark[i].x = results.left_hand_landmarks.landmark[i].x
                    results.left_hand_landmarks.landmark[i].y = results.left_hand_landmarks.landmark[i].y
                    results.left_hand_landmarks.landmark[i].z = results.left_hand_landmarks.landmark[i].z
                    left_hand_data.update(
                        {left_hand_points[i]: results.left_hand_landmarks.landmark[i]}
                    )

                keys = []
                for key in left_hand_data.keys():
                    keys.append(key)

                left_hand_xyz_data = np.empty((21, 3))

                index = 0
                for value in left_hand_data.values():
                    x = value.x
                    y = value.y
                    z = value.z

                    # append to numpy array
                    left_hand_xyz_data[index] = [x, y, z]

                    index += 1

            else:
                left_hand_xyz_data = np.empty((21, 3))
                left_hand_xyz_data[:] = np.nan

            # right hand landmarks
            if results.right_hand_landmarks:
                right_hand_data = {}
                for i in range(len(right_hand_points)):
                    results.right_hand_landmarks.landmark[i].x = results.right_hand_landmarks.landmark[i].x
                    results.right_hand_landmarks.landmark[i].y = results.right_hand_landmarks.landmark[i].y
                    results.right_hand_landmarks.landmark[i].z = results.right_hand_landmarks.landmark[i].z
                    right_hand_data.update(
                        {right_hand_points[i]: results.right_hand_landmarks.landmark[i]}
                    )
                keys = []
                for key in right_hand_data.keys():
                    keys.append(key)

                right_hand_xyz_data = np.zeros((21, 3))

                index = 0
                for value in right_hand_data.values():
                    x = value.x
                    y = value.y
                    z = value.z

                    # append to numpy array
                    right_hand_xyz_data[index] = [x, y, z]

                    index += 1
            else:
                right_hand_xyz_data = np.empty((21, 3))
                right_hand_xyz_data[:] = np.nan

            left_hand.append(left_hand_xyz_data)
            right_hand.append(right_hand_xyz_data)
            pose.append(pose_xyz_data)

            frames += 1

    cap.release()
    # cv2.destroyAllWindows()
    left_hand = np.array(left_hand)
    right_hand = np.array(right_hand)
    pose = np.array(pose)

    # frame interpolation
    # left hand landmarks
    if np.isnan(left_hand).any:
    # interpolate
        if np.count_nonzero(np.isnan(left_hand)) == left_hand.shape[0]*left_hand.shape[1]*left_hand.shape[2]:
            # zero-based interpolation
            left_hand = np.zeros((left_hand.shape[0], left_hand.shape[1], left_hand.shape[2]))
        else:
            # linear interpolation
            left_hand = interpolate_pose_frames(left_hand, body="left")
            print("interpolating left hand")
    # right hand landmarks
    if np.isnan(right_hand).any:
        if np.count_nonzero(np.isnan(right_hand)) == right_hand.shape[0]*right_hand.shape[1]*right_hand.shape[2]:
            # zero-based interpolation
            right_hand = np.zeros((right_hand.shape[0], right_hand.shape[1], right_hand.shape[2]))
        else:
            # linear interpolation
            right_hand = interpolate_pose_frames(right_hand, body="right")
            print("interpolating right hand")
    # pose landmarks
    if np.isnan(pose).any:
        if np.count_nonzero(np.isnan(pose)) == pose.shape[0]*pose.shape[1]*pose.shape[2]:
            # zero-based interpolation
            pose = np.zeros((pose.shape[0], pose.shape[1], pose.shape[2]))
        else:
            # linear interpolation
            pose = interpolate_pose_frames(pose, body="pose")
            print("interpolating whole body pose")

    left_hand_list = left_hand.tolist()
    right_hand_list = right_hand.tolist()
    pose_list = pose.tolist()

    # concatenate all landmarks subsets
    for frame in range(frames):
        all_frame_landmarks = left_hand_list[frame] + right_hand_list[frame] + pose_list[frame]
        all_video_landmarks.append(all_frame_landmarks)

    all_video_landmarks = np.array(all_video_landmarks)

    return all_video_landmarks, frames


def main():
    landmark_names = return_all_landmark_names()
    video_landmarks, frames = generate_video_landmarks("032_004_001.mp4")
    print(video_landmarks)

    print(len(video_landmarks))
    print(video_landmarks.shape)

if __name__ == "__main__":
    main()
