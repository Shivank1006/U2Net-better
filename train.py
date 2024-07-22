import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

from data_loader import SalObjDataset, RescaleT, RandomCrop, ToTensorLab
from model import U2NET, U2NETP

# Define loss function
def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    bce_loss = nn.BCELoss(reduction='mean')
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print(f"l0: {loss0.item():.3f}, l1: {loss1.item():.3f}, l2: {loss2.item():.3f}, "
          f"l3: {loss3.item():.3f}, l4: {loss4.item():.3f}, l5: {loss5.item():.3f}, "
          f"l6: {loss6.item():.3f}")

    return loss0, loss

# Set directories and parameters
model_name = 'u2net' 
data_dir = os.path.join(os.getcwd(), 'train_data')
tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug')
tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug')

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name)
os.makedirs(model_dir, exist_ok=True)

epoch_num = 100000
batch_size_train = 12
batch_size_val = 1

# Prepare data
tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, f'*{image_ext}'))
tra_lbl_name_list = [os.path.join(data_dir, tra_label_dir, 
                     os.path.splitext(os.path.basename(img_path))[0] + label_ext)
                     for img_path in tra_img_name_list]

print(f"Training images: {len(tra_img_name_list)}")
print(f"Training labels: {len(tra_lbl_name_list)}")

# Define transforms and datasets
train_transform = transforms.Compose([
    RescaleT(320),
    RandomCrop(288),
    ToTensorLab(flag=0)
])

val_transform = transforms.Compose([
    RescaleT(320),
    ToTensorLab(flag=0)
])

train_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=train_transform)

# Assuming you have validation data
val_dataset = SalObjDataset(
    img_name_list=val_img_name_list,
    lbl_name_list=val_lbl_name_list,
    transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size_train, 
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size_val, 
                        shuffle=False, num_workers=4, pin_memory=True)

# Define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = U2NET(3, 1) if model_name == 'u2net' else U2NETP(3, 1)
net.to(device)

# Define optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), 
                       eps=1e-08, weight_decay=0)

# Initialize wandb
wandb.init(project="u2net_salient_object_detection", name="u2net_training")
wandb.config.update({
    "learning_rate": 0.001,
    "epochs": epoch_num,
    "batch_size": batch_size_train,
    "model": model_name,
})

# Validation function
def validate(net, dataloader):
    net.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data['image'].to(device), data['label'].to(device)
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Training loop
for epoch in range(epoch_num):
    net.train()
    epoch_loss = 0
    epoch_tar_loss = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data['image'].to(device), data['label'].to(device)
        
        optimizer.zero_grad()

        d0, d1, d2, d3, d4, d5, d6 = net(inputs)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_tar_loss += loss2.item()

        # Log metrics
        wandb.log({
            "iteration": i + epoch * len(train_loader),
            "train_loss": loss.item(),
            "train_tar_loss": loss2.item(),
        })

    # Log epoch metrics
    wandb.log({
        "epoch": epoch + 1,
        "epoch_loss": epoch_loss / len(train_loader),
        "epoch_tar_loss": epoch_tar_loss / len(train_loader),
    })

    # Validation step
    if (epoch + 1) % 5 == 0:
        val_loss = validate(net, val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "val_loss": val_loss,
        })

    # Save model
    if (epoch + 1) % 10 == 0:
        torch.save(net.state_dict(), f"{model_dir}/{model_name}_epoch_{epoch+1}.pth")
        wandb.save(f"{model_dir}/{model_name}_epoch_{epoch+1}.pth")

    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

# Finish the wandb run
wandb.finish()