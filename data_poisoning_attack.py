# pytorch libs
from tqdm import tqdm
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

from poison_dataset import add_poison

torch.manual_seed(42)
torch.cuda.manual_seed(42)
random_seed = 42


# wandb.login(key=config.API_KEY)
# print("[LOG]: Login Succesfull.")


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] current used device: {device}")


# Getting DATASET

# defining transform
transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


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

test_dataloader_map = {}
poisoned_test_dataloader_map = {}

# converting data into torch dataloader
import os
BATCH_SIZE = 64
NUM_WORKERS = 4

print("[INFO] Creating dataloader for test dataset splits")
print("[INFO] Creating Poisones dataloader for test dataset splits")
for test_split in tqdm(test_dataset_splits, total=len(test_dataset_splits)):
    # print(f"[INFO] test_split: {test_split}")
    cur_test_dataset = test_dataset_splits[test_split]
    cur_test_dataset_dataloader = DataLoader(
        cur_test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    test_dataloader_map[test_split] = cur_test_dataset_dataloader
    poisoned_test_dataset = add_poison(cur_test_dataset,101)
    poisoned_test_dataloader = DataLoader(
        poisoned_test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    poisoned_test_dataloader_map[test_split] = poisoned_test_dataloader

print("[INFO] Dataloader for test dataset splits created successfully.")


# Testing one batch from train dataloader
# print("\n\n1st batch form train dataloader")
# print(next(iter(train_dataloader))[1])

# Importing model
from models import Resnet18, Resnet50, EfficentNetB0, MobileNetV2, VITBase16

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
# Training model on train data
from engine import testing_step, train, training_step
from timeit import default_timer as timer 

# Hyperparms
model = get_resnet_18_model().to(device)
model.load_state_dict(torch.load("./models/Resnet18_epoch_100_optim_adam_lr_0.001_betas_(0.8, 0.888)_eps_1e-08_weight_decay_0.001.pth"))
model.eval()
print("[INFO] Model loaded successfully.")
print(f"[INFO] Model name: {model.__class__.__name__}")


attack_detected,attact_flag_count,prev_acc = False,0,1.0
learning_rate = 0.001
betas = (0.8,0.888)
eps = 1e-08
weight_decay = 0.001
acc_fn = Accuracy("multiclass",num_classes=101).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)


print("[INFO] Strating Attack!!!")

# fucntion to detect attacks
def detect_for_attacks(acc):
    global attack_detected,attact_flag_count,prev_acc
    if acc < prev_acc:
        attact_flag_count += 1
    else:
        attact_flag_count = 0
    if attact_flag_count >= 3:
        attack_detected = True
    prev_acc = acc
    
# function to get overall loss and acc
def get_overall_loss_and_acc():
    loss,acc = 0,0
    for dataloader_split in test_dataloader_map:
        test_dataloader = test_dataloader_map[dataloader_split]
        l,a = testing_step(model=model,dataloader=test_dataloader,loss_fn=loss_fn,acc_fn=acc_fn,device=device)
        loss+=l
        acc+=a
    return loss/len(test_dataloader_map),acc/len(test_dataloader_map)

for dataloader_split in test_dataloader_map:
    
    test_dataloader = test_dataloader_map[dataloader_split]
    poisoned_test_dataloader = poisoned_test_dataloader_map[dataloader_split]
    
    print(f"[INFO] test_split: {dataloader_split}")
    print("BEFORE ONLINE LEARNING")
    loss, acc = get_overall_loss_and_acc()
    print(f"[INFO] Overall loss: {loss:.4f}, acc: {acc:.4f}")
    
    for epoch in tqdm(range(1,11), total=10, desc="traing on poisoned data"):
        loss, acc = training_step(model=model,dataloader=poisoned_test_dataloader,loss_fn=loss_fn,acc_fn=acc_fn,optimizer=optimizer,device=device)
        print(f"[LOG] loss: {loss:.4f}, acc: {acc:.4f}")
    
    print("AFTER ONLINE LEARNING WITH POISONED DATA")
    loss, acc = 0,0
    loss, acc = get_overall_loss_and_acc()
    print(f"[INFO] Overall loss: {loss:.4f}, acc: {acc:.4f}")
    detect_for_attacks(acc)
    if not attack_detected:
        print("[INFO] Attack Detected!!!, Please review the batch of the data.")
    
print("[INFO] Attack Finished!!!")