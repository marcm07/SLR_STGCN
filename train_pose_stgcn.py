"""
Model training script of ST-GCN pose based sign language recognition
Author: Marc Marais
"""
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset
from dataset_loaders.video_dataset_skeletal import VideoFrameDataset, PoselistToTensorV2
import torch
import os
from torchvision import transforms
from pytorchtools import EarlyStopping, plot_loss_curve, plot_accuracy_curve
import wandb
from sklearn import metrics
import pandas as pd
import numpy as np
from models.stgcn import STGCN
from graphs.mpg import MediapipeGraph
from torchviz import make_dot
from torchview import draw_graph
torch.set_default_dtype(torch.double)

# initialise wandb
wandb.login(key="49191acb101a06a98c22cdb828e90bdb24fd33fb")

# initialise model run parameters
BATCH_SIZE =  4
accumulation_steps = 8
effective_bs = BATCH_SIZE*accumulation_steps
epochs = 300
lr = 5e-5
gamma = 0.5
seed = 42
NUM_FRAMES = 16
image_size = 256
patience = 20
num_classes = 226
dataset = "AUTSL"
save_loc = "output/run1_2plus_1d_autsl"

if not os.path.exists(save_loc):
    os.makedirs(save_loc)

wandb.init(
    project="Pose_Estimation_GCNs",
    name="mpPose_stgcn_16frames",
    config={
        "learning_rate": lr,
        "epochs": epochs,
        "Batch_size": effective_bs,
        "num_frames": NUM_FRAMES,
        "dataset": dataset,
        "model": "stgcn_full_dropout",
    }
)

# initialise root dir of dataset in use
if dataset == "AUTSL":
    root_dir = "Data_Pose/AUTSL/"
else:
    root_dir = "Data_Pose/LSA64/"

device = 'cuda'

preprocess = transforms.Compose([
        PoselistToTensorV2()
    ])

# Dataset Loading

# augment training set
train_preprocess = transforms.Compose([
    PoselistToTensorV2(),
#     augmentations for visual data
#     transforms.RandomApply(torch.nn.ModuleList([
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation((-3, 3)),
#         transforms.RandomAffine(degrees=(-3, 3), translate=(0.0, 0.33), scale=(0.8, 1.1), shear=(-3, 3)),
#     ]), p=0.7)
])

train_videos_root = os.path.join(root_dir, "train")
print(train_videos_root)
train_annotation_file = os.path.join(train_videos_root, 'train_annotations.txt')

train_dataset = VideoFrameDataset(
    root_path=train_videos_root,
    annotationfile_path=train_annotation_file,
    num_segments=NUM_FRAMES,
    frames_per_segment=1,
    imagefile_template='img_{:05d}.jpg',
    transform=train_preprocess
)

print(len(train_dataset))

val_videos_root = os.path.join(root_dir, "val")
val_annotation_file = os.path.join(val_videos_root, 'val_annotations.txt')

val_dataset = VideoFrameDataset(
    root_path=val_videos_root,
    annotationfile_path=val_annotation_file,
    num_segments=NUM_FRAMES,
    frames_per_segment=1,
    imagefile_template='img_{:05d}.jpg',
    transform=train_preprocess,
)
print(len(val_dataset))
# %%
test_videos_root = os.path.join(root_dir, "test")
test_annotation_file = os.path.join(test_videos_root, 'test_annotations.txt')

test_dataset = VideoFrameDataset(
    root_path=test_videos_root,
    annotationfile_path=test_annotation_file,
    num_segments=NUM_FRAMES,
    frames_per_segment=1,
    imagefile_template='img_{:05d}.jpg',
    transform=preprocess,
)
print(len(test_dataset))
# %%
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=False
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=False
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=False
)

# check dataloader
for epoch in range(1):
    for video_batch, labels in train_dataloader:
        """
        Insert Training Code Here
        """
        print(labels)
        print("\nVideo Batch Tensor Size:", video_batch.size())
        print("Batch Labels Size:", labels.size())
        break
    break



node_features = 3 # x,y,z
nodes = 59

model = STGCN(num_class=num_classes,
                   num_point=nodes,
                    num_person=1,
                  graph_type=MediapipeGraph(),
                  in_channels=3,
                  cuda_= True).to(device)

model = model.float()
# model.load_state_dict(torch.load(save_loc + "checkpoint_autsl_pose_run1_16.pt")['model_state_dict'])
print(model)

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
# scheduler
scheduler = StepLR(optimizer, step_size=15, gamma=gamma)


def train_model(model, patience, epochs, accumulation_steps=1):
    """
    Train the model with early stooping and gather and report metrics locally and to wandb
    :param model: pytorch model being trained
    :param patience (int): early stopping patience
    :param epochs (int): number of epochs to iterate over
    :return: model and model evaluation metrics
    """
    # model data to log
    avg_train_losses = []
    avg_val_losses = []
    train_accuracies = []
    val_accuracies = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()
        for batch_idx, (data, label) in enumerate(train_dataloader):

            # extract inputs and labels
            data = data.to(device)
            label = label.to(device)

            # forward pass
            output = model(data.float())
            loss = criterion(output, label)

            # normalize loss to account for batch accumulation
            loss = loss / accum_iter

            # backward pass
            loss.backward()

            # weights update
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_dataloader)
            epoch_loss += loss / len(train_dataloader)

        model.eval()
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in val_dataloader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data.float())
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(val_dataloader)
                epoch_val_loss += val_loss / len(val_dataloader)

        avg_train_losses.append(epoch_loss.detach().cpu())
        avg_val_losses.append(epoch_val_loss.detach().cpu())
        train_accuracies.append(epoch_accuracy.detach().cpu())
        val_accuracies.append(epoch_val_accuracy.detach().cpu())

        epoch_len = len(str(epochs))

        print_msg = (f'[{(epoch + 1):>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {epoch_loss:.5f} ' + f'train_acc: {epoch_accuracy:.5f} ' +
                     f'val_loss: {epoch_val_loss:.5f} ' + f'val_acc: {epoch_val_accuracy:.4f} ')

        print(print_msg)

        wandb.log({
            "Epoch": epoch,
            "Train Loss": epoch_loss,
            "Train Acc": epoch_accuracy,
            "Valid Loss": epoch_val_loss,
            "Valid Acc": epoch_val_accuracy})

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss=epoch_val_loss, model=model, path=save_loc + "checkpoint_autsl_pose_run1_16.pt",
                       EPOCH=epoch,
                       train_loss=epoch_loss, optimizer=optimizer)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(save_loc + "checkpoint_autsl_pose_run1_16.pt")['model_state_dict'])

    # visualize the loss as the network trained
    # loss
    plot_loss_curve(avg_train_losses, avg_val_losses, save_loc, "loss_plot_01.png")
    # accuracy plot
    plot_accuracy_curve(train_accuracies, val_accuracies, avg_val_losses, save_loc, "accuracy_plot_01.png")

    # save entire trained model
    # torch.save(model, os.path.join(save_loc, "lsa64_pose_lr1e-4_16.pt"))

    # save model weights
    torch.save(model.state_dict(), os.path.join(save_loc, "lsa64_pose_lr1e-4_16_weights.pt"))

    return model, avg_train_losses, avg_val_losses, train_accuracies, val_accuracies

# train model
model, train_loss, valid_loss, train_accuracies, val_accuracies = train_model(model, patience, epochs, accumulation_steps)

# model testing
correct = 0
total = 0
y_true = []
y_pred = []
model.eval()
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data, labels in test_dataloader:
        data = data.to(device)
        labels = labels.to(device)

        # calculate outputs by running images through the network
        outputs = model(data.float())
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        
        # append to y_true and y_pred
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# print model results
print(f'Accuracy of the network: {100 * correct // total} %')
accuracy_score = metrics.accuracy_score(y_true, y_pred)
print(f"Accuracy sklearn: {accuracy_score:.4f}")
f_score = metrics.f1_score(y_true, y_pred, average="weighted")
print(f"F-score: {f_score:.4f}")
balanced_accuracy_score = metrics.balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy sklearn: {balanced_accuracy_score:.4f}")

lsa64_classes = ["Opaque", "Red", "Green", "Yellow", "Bright", "Light_blue", "Colors", "Red2", "Women", "Enemy", "Son", "Man",
           "Away", "Drawer", "Born", "Learn", "Call", "Skimmer", "Bitter", "Sweet_milk", "Milk", "Water", "Food", "Argentina",
           "Urugauy", "Country", "Last_name", "Where", "Mock", "Birthday", "Breakfast", "Photo", "Hungry", "Map",
           "Coin", "Music", "Ship", "None", "Name", "Patience", "Perfume", "Deaf", "Trap", "Rice", "Barbecue",
           "Candy", "Chewing_gum", "Spaghetti", "Yogurt", "Accept", "Thanks", "Shut_down", "Appear", "To_land",
           "Catch", "Help", "Dance", "Bathe", "Buy", "Copy", "Run", "Realize", "Give", "Find"]

autsl_classes = pd.read_csv("../LSA64_pytorch/AUTSL/SignList_ClassId_TR_EN.csv")
autsl_class_names = np.array(pd.unique(autsl_classes['EN']))
for index in range(len(autsl_class_names)):
    autsl_class_names[index] = autsl_class_names[index].replace(' ', '_')

# print classification report of classes
if dataset == "AUTSL":
    print(metrics.classification_report(y_true, y_pred, target_names=autsl_class_names))
else:
    print(metrics.classification_report(y_true, y_pred, target_names=lsa64_classes))

