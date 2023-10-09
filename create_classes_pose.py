import cv2
import shutil
import os
import splitfolders
import pandas as pd
import numpy as np
from mp_landmarks_gen_video import generate_video_landmarks

classes = ["Opaque", "Red", "Green", "Yellow", "Bright", "Light_blue", "Colors", "Red2", "Women", "Enemy", "Son", "Man",
           "Away", "Drawer", "Born", "Learn", "Call", "Skimmer", "Bitter", "Sweet_milk", "Milk", "Water", "Food", "Argentina",
           "Urugauy", "Country", "Last_name", "Where", "Mock", "Birthday", "Breakfast", "Photo", "Hungry", "Map",
           "Coin", "Music", "Ship", "None", "Name", "Patience", "Perfume", "Deaf", "Trap", "Rice", "Barbecue",
           "Candy", "Chewing_gum", "Spaghetti", "Yogurt", "Accept", "Thanks", "Shut_down", "Appear", "To_land",
           "Catch", "Help", "Dance", "Bathe", "Buy", "Copy", "Run", "Realize", "Give", "Find"]

first_5_classes = ["Opaque", "Red", "Green", "Yellow"]

autsl_classes = pd.read_csv("SignList_ClassId_TR_EN.csv")
print(len(pd.unique(autsl_classes['EN'])))
print(len(autsl_classes['EN']))
print(autsl_classes.shape)
autsl_class_names = np.array(pd.unique(autsl_classes['EN']))
print(autsl_class_names)
# autsl_class_names = autsl_class_names.replace(' ', '_')
for index in range(len(autsl_class_names)):
    autsl_class_names[index] = autsl_class_names[index].replace(' ', '_')
print(autsl_class_names)

def create_class_folders():
    if os.path.exists("Sign_Classes"):
        shutil.rmtree("Sign_Classes")
    for sign_class in first_5_classes: #change for full dataset
        os.makedirs("Sign_Classes/" + str(sign_class))

def get_video_class(file_name):
    return int(file_name[1:3])

def write_frames_to_class(frames, dir):
    index = 0
    for frame in frames:
        # dir = "Sign_Classes/" + first_3_classes[sign_class] + "/" + "Video" + str(video_num)
        if os.path.exists(dir) == False:
            os.makedirs(dir)
        file_name = dir + "/" + "img_" + f"{index:05}" + ".jpg"
        cv2.imwrite(file_name, frame)
        # print(file_name)
        index += 1
def resize_frame(frame, width=None, height=None, inter = cv2.INTER_AREA):
    # get original image dimensions
    dim = None
    # if both the width and height are None, then return the
    # original frames

    (h, w) = frame.shape[:2]
    if width is None and height is None:
        return frame

    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))
    resized_frame = cv2.resize(frame, dim, interpolation=inter)

    return resized_frame

def center_crop(img, dim=(256,256)):
    """Returns center cropped image

    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    width, height = img.shape[1], img.shape[0]  #process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

def extract_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    frames = []
    index = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            # print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        # if index % 5 == 0: #only store every 20th frame
        image = resize_frame(image, height=512)
#         image = center_crop(image, dim=(256,256))
        # image = resize_frame(image, height=256, width=256)
        frames.append(image)
        index += 1
    # print(index)
    cap.release()
    return frames, index

def iterate_through_videofiles(path):
    for dirname, _, filenames in sorted(os.walk(path)):
        count = 0
        filenames.sort()
        for filename in filenames:
            class_num = get_video_class(filename)-1
            frames = extract_frames(os.path.join(dirname, filename))
            write_frames_to_class(frames, filename, class_num)
            print(count)
            count += 1

def sort_videos(path):
    for dirname, _, filenames in sorted(os.walk(path)):
        count = 0
        filenames.sort()
        for filename in filenames:
            # class_num = get_video_class(filename) - 1
            new_dir = "Inceptionv3_weights/Sign_Images_Signer_Independent/" + filename
            old_dir = os.path.join(dirname, filename)
            shutil.copy(old_dir, new_dir)
            print(count)
            count += 1

def video_class_folders(path, class_names):
    splits = ['train', 'val', 'test']
    if os.path.exists(path):
        shutil.rmtree(path)
    for split in splits:
        for sign_class in class_names: #change for full dataset
            os.makedirs(path +"/" + split + "/" + str(sign_class))

def create_train_val_test_splits(input_path, output_path):
    train = 0.8
    val = 0
    test = 0.2
    splitfolders.ratio(input_path, output=output_path, ratio=(train, val, test))

def get_signer_number(filename):
    return int(filename[4:7])

def signer_independence():
    path = "Signer_Independent_Splits"
    if os.path.exists(path):
        shutil.rmtree(path)
    splits = ['train', 'val', 'test']
    for split in splits:
            os.makedirs(path + "/" + split)

    video_path = "lsa64_raw/all"

    for dirname, _, filenames in sorted(os.walk(video_path)):
        count = 0
        filenames.sort()
        for filename in filenames:
            # class_num = get_video_class(filename)
            # print(class_num)
            signer_num = get_signer_number(filename)
            if signer_num==5 or signer_num==10:
                split = 'test'
            elif signer_num==7:
                split = 'val'
            else:
                split = 'train'
            new_dir = path + "/" + split + "/" + filename
            print(new_dir)
            old_dir = os.path.join(dirname, filename)
            shutil.copy(old_dir, new_dir)
            print(count)
            count += 1

def get_autsl_video_class(filename, labels):
    """"
    Extracts the class number from the labels dataframe by locating the video name

    args
    filename (str): name of video file
    labels: dataframe containing video names and class numbers
    """
    video_name = filename.split('_')[0] + "_" + filename.split('_')[1]
    video_row = labels.loc[labels['Signer'] == video_name]
    return video_row["Class"].values[0]

def save_pose_frames(video_landamrks, dir):
    index = 0
    for frame in video_landamrks:
        if os.path.exists(dir) == False:
            os.makedirs(dir)
        file_name = dir + "/" + "img_" + f"{index:05}" + ".npy"
        np.save(file_name, frame)
        index += 1

def generate_sign_images_autsl(dir, labels_file, annotations_file, split, root_dir):
    # file to store video_path, start_frame, end_frame, class_index
    annotations_file = open(root_dir + '/' + split + '/' + annotations_file, 'w')
    #load csv labels file
    labels = pd.read_csv(labels_file)

    root_dir = root_dir + '/' + split
    for dirname, _, filenames in os.walk(dir):
        for filename in filenames:
            class_num = get_autsl_video_class(filename, labels)
            class_name = autsl_class_names[class_num]

            video_landmarks, frames = generate_video_landmarks(os.path.join(dirname, filename))
            end_frame = frames
            # save org frames
            new_dir = class_name + "/" + filename.split('.')[0]
            print(new_dir)

            write_frames_to_class(frames, os.path.join(root_dir, new_dir))

            save_pose_frames(video_landmarks, os.path.join(root_dir, new_dir))

            # update annotations file
            annotations_file.write(str(new_dir))
            annotations_file.write(' ')
            annotations_file.write(str(1))
            annotations_file.write(' ')
            annotations_file.write(str(end_frame-1))
            annotations_file.write(' ')
            annotations_file.write(str(class_num))
            annotations_file.write('\n')

    annotations_file.close()


def generate_sign_images_LSA64(dir, annotations_file, split, root_dir):
    # file to store video_path, start_frame, end_frame, class_index
    annotations_file = open(root_dir + '/' + split + '/' + annotations_file, 'w')
    root_dir = root_dir + '/' + split
    for dirname, _, filenames in os.walk(dir):
        for filename in filenames:
            # get class number from video filename
            class_num = get_video_class(filename)
            class_name = classes[class_num-1]
            class_num = int(class_num) - 1

            video_landmarks, frames = generate_video_landmarks(os.path.join(dirname, filename))
            print(video_landmarks.shape)

            end_frame = frames
            # save org frames
            new_dir = class_name + "/" + filename.split('.')[0]
            print(new_dir)
            write_frames_to_class(frames, os.path.join(root_dir, new_dir))
            save_pose_frames(video_landmarks, os.path.join(root_dir, new_dir))
            annotations_file.write(str(new_dir))
            annotations_file.write(' ')
            annotations_file.write(str(1))
            annotations_file.write(' ')
            annotations_file.write(str(end_frame - 1))
            annotations_file.write(' ')
            annotations_file.write(str(class_num))
            annotations_file.write('\n')

    annotations_file.close()

def main():
    #LSA64 dataset
    video_class_folders("Data_Pose/", class_names=classes)
    print("Train pose points...")
    generate_sign_images_LSA64("Signer_Independent_Splits/train",
                               annotations_file="train_annotations.txt", root_dir="Data_Pose/",
                               split='train')
    print("Val pose points...")
    generate_sign_images_LSA64("Signer_Independent_Splits/val",
                               annotations_file="val_annotations.txt", root_dir="Data_Pose/",
                               split='val')
    print("Test pose points...")
    generate_sign_images_LSA64("Signer_Independent_Splits/test",
                               annotations_file="test_annotations.txt", root_dir="Data_Pose/",
                               split='test')
    print("Done")
    
    # autsl dataset
    video_class_folders("AUTSL/Pose_Data", class_names=autsl_class_names)
    generate_sign_images_autsl("/mnt/Massive_bk1/marc_bk/AUTSL/Data/RGB/train", labels_file="AUTSL/train_labels.csv",
                               annotations_file="train_annotations.txt", root_dir="AUTSL/Data",
                               split='train')
    print("training images done")
    generate_sign_images_autsl("/mnt/Massive_bk1/marc_bk/AUTSL/Data/RGB/val", labels_file="AUTSL/val_labels.csv",
                               annotations_file="val_annotations.txt", root_dir="AUTSL/Data",
                               split='val')
    print("val images done")
    generate_sign_images_autsl("./mnt/Massive_bk1/marc_bk/AUTSL/Data/RGB/test",  labels_file="AUTSL/test_labels.csv",
                               annotations_file="test_annotations.txt", root_dir="AUTSL/Pose_Data",
                               split='test')
    print("Done")


if __name__ == "__main__":
    main()
