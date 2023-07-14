# pytorch libs

from plotting import plot_curves
import torch
from torch import nn
import torchvision
import os

# numpy
import numpy as np

# torch metrics

from torchmetrics import Accuracy

from torch.utils.data import DataLoader

from torchvision import transforms
import wandb
import config

torch.manual_seed(42)
torch.cuda.manual_seed(42)
random_seed = 42


wandb.login(key=config.API_KEY)
print("[LOG]: Login Succesfull.")


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] current used device: {device}")


# Getting DATASET

# defining transform
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.TrivialAugmentWide(num_magnitude_bins=3),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


train_dataset = torchvision.datasets.Food101(
    download=True,
    root='./data',
    split='train',
    transform=transform_train
)


test_dataset = torchvision.datasets.Food101(
    download=True,
    root='./data',
    split='test',
    transform=transform_test
)

# print(f"[INFO] dataset size: {len(dataset)}")
# print(f"[INFO] dataset classes length: {len(dataset.classes)}")
# print(f"[INFO] dataset class to idx mapping: {dataset.class_to_idx}")
# print(f"[INFO] datset[0] shape: {dataset[0][0].shape}")

# Creating subset of test_datasets such that, can used as online data
test_dataset_splits = {}
SPLITS = 10
samples_per_split = int(len(test_dataset)/SPLITS)

test_dataset_1, test_dataset_2, test_dataset_3, test_dataset_4, test_dataset_5, test_dataset_6, test_dataset_7, test_dataset_8, test_dataset_9, test_dataset_10 = torch.utils.data.random_split(test_dataset, [samples_per_split]*SPLITS)
test_dataset_splits["test_dataset_1"] = test_dataset_1
test_dataset_splits["test_dataset_2"] = test_dataset_2
test_dataset_splits["test_dataset_3"] = test_dataset_3
test_dataset_splits["test_dataset_4"] = test_dataset_4
test_dataset_splits["test_dataset_5"] = test_dataset_5
test_dataset_splits["test_dataset_6"] = test_dataset_6
test_dataset_splits["test_dataset_7"] = test_dataset_7
test_dataset_splits["test_dataset_8"] = test_dataset_8
test_dataset_splits["test_dataset_9"] = test_dataset_9
test_dataset_splits["test_dataset_10"] = test_dataset_10


# converting data into torch dataloader
import os
BATCH_SIZE = 64
NUM_WORKERS = 4

train_dataloader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_dataloader = DataLoader(
    test_dataset_splits["test_dataset_1"],
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# Testing one batch from train dataloader
# print("\n\n1st batch form train dataloader")
# print(next(iter(train_dataloader))[1])

# Importing model
from models import EarlyStopping, Resnet18, Resnet50, EfficentNetB0, MobileNetV2, VITBase16

def get_resnet_18_model():
    model = Resnet18(out_shape=101)
    return model


def get_resnet_50_model():
    model = Resnet50(out_shape=101)
    return model


def get_effnet_B0_model():
    model = EfficentNetB0(out_shape=101)
    return model

def get_mobilenet_v2_model():
    model = MobileNetV2(out_shape=101)
    return model


def get_vit_base_16_model():
    model = VITBase16(out_shape=101)
    return model




# Train Info
# Early stopping
early_stopping = EarlyStopping(tolerance=3, min_delta=0.001)

# Training model on train data
from engine import train
from timeit import default_timer as timer 

# Hyperparms
lr = [1e-3,1e-4] # learning rate
betas=[(0.8, 0.888)] # coefficients used for computing running averages of gradient and its square
eps = [1e-8] # term added to the denominator to improve numerical stability
weight_decay = [1e-3] # weight decay (L2 penalty)

# init. epochs
NUM_EPOCHS = [100]

parms_combs = [(l,b,e,w_d,epochs) for l in lr for b in betas for e in eps for w_d in weight_decay for epochs in NUM_EPOCHS]

# init. loss function, accuracy function and optimizer
loss_fn = nn.CrossEntropyLoss()
acc_fn = Accuracy(task="multiclass", num_classes=101).to(device=device)

cur,total = 1, len(lr)*len(betas)*len(eps)*len(weight_decay)*len(NUM_EPOCHS)
for h_parms in parms_combs:
    wandb.init(
        # set the wandb project where this run will be logged
        project="dai-project",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": h_parms[0],
        "weight_decay": h_parms[3],
        "architecture": "ViT Base 16",
        "dataset": "FOOD-101",
        "epochs": h_parms[4],
        "batch_size": BATCH_SIZE,
        "seed_value": 42,
        }
    )

    ### INIT MODEL STARTS ###
    # traning same model for each parms
    model = get_vit_base_16_model().to(device=device)
    ### INIT MODEL END ###

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=h_parms[0], betas=h_parms[1], eps=h_parms[2],weight_decay=h_parms[3]
    )

    # importing and init. the timer for checking model training time
    from timeit import default_timer as timer

    start_time = timer()
    print(f"current exp / total: {cur} / {total}")
    print(f"Training with: lr: {h_parms[0]}, betas: {h_parms[1]}, eps: {h_parms[2]}, weight_decay: {h_parms[3]}")

    model_results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        epochs=h_parms[4],
        save_info=f"lr_{h_parms[0]}_betas_{h_parms[1]}_eps_{h_parms[2]}_weight_decay_{h_parms[3]}",
        device=device
    )

    # end timer
    end_time = timer()
    # printing time taken
    print(f"total training time: {end_time-start_time:.3f} sec.")
    # print("model stats:")
    # print(model_0_results)
    print(f"LOSS & Accuracy Curves\n"
        f"lr: {h_parms[0]}, betas: {h_parms[1]}, eps: {h_parms[2]}, weight_decay: {h_parms[3]}")
    plot_curves(model_results,f"{model.__class__.__name__}_epoch_{h_parms[4]}_optim_adam_"
                +
                f"lr_{h_parms[0]}_betas_{h_parms[1]}_eps_{h_parms[2]}_weight_decay_{h_parms[3]}")
    cur+=1
    print()