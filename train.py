import os
import re
import glob
import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP

# Configuration
CONFIG = {
    "project_name": "u2net_training",
    "model_name": "u2net",  # or "u2netp"
    "epochs": 100,
    "batch_size_train": 8,
    "batch_size_val": 4,
    "learning_rate": 0.001,
    "save_freq": 1,
    "data_dir": 'preprocessed_dataset',
    "model_dir": 'saved_models',
    "resolution": 512,
    "resume_training": True 
}


def setup_data(data_dir: str) -> Tuple[DataLoader, DataLoader]:
    tra_image_dir = os.path.join('train', 'images')
    tra_label_dir = os.path.join('train', 'masks')
    test_image_dir = os.path.join('test', 'images')
    test_label_dir = os.path.join('test', 'masks')

    image_ext = '.jpg'
    label_ext = '.jpg'

    tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, f'*{image_ext}'))
    tra_lbl_name_list = [os.path.join(data_dir, tra_label_dir, os.path.splitext(os.path.basename(img_path))[0] + label_ext)
                         for img_path in tra_img_name_list]

    test_img_name_list = glob.glob(os.path.join(data_dir, test_image_dir, f'*{image_ext}'))
    test_lbl_name_list = [os.path.join(data_dir, test_label_dir, os.path.splitext(os.path.basename(img_path))[0] + label_ext)
                          for img_path in test_img_name_list]

    train_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(CONFIG['resolution']),
            ToTensorLab(flag=0)
        ])
    )

    test_dataset = SalObjDataset(
        img_name_list=test_img_name_list,
        lbl_name_list=test_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(CONFIG['resolution']),
            ToTensorLab(flag=0)
        ])
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size_train'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size_val'], shuffle=False, num_workers=4)

    return train_loader, test_loader


def create_model(model_name: str) -> nn.Module:
    if model_name == "u2net":
        return U2NET(3, 1)
    elif model_name == "u2netp":
        return U2NETP(3, 1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


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
    return loss0, loss


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_tar_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        inputs, labels = batch['image'].to(device).float(), batch['label'].to(device).float()
        optimizer.zero_grad()
        d0, d1, d2, d3, d4, d5, d6 = model(inputs)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_tar_loss += loss2.item()
    return total_loss / len(dataloader), total_tar_loss / len(dataloader)


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float, List]:
    model.eval()
    total_loss = 0.0
    total_tar_loss = 0.0
    wandb_images = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['image'].to(device).float(), batch['label'].to(device).float()
            d0, d1, d2, d3, d4, d5, d6 = model(inputs)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)
            total_loss += loss.item()
            total_tar_loss += loss2.item()

            # Prepare images for wandb logging
            if len(wandb_images) < 5:  # Limit to 5 images
                for j in range(len(inputs)):
                    wandb_images.append([
                        wandb.Image(inputs[j].cpu().numpy().transpose(1, 2, 0), caption="Image"),
                        wandb.Image(labels[j].cpu().numpy().transpose(1, 2, 0), caption="Ground Truth"),
                        wandb.Image(d0[j].cpu().numpy().transpose(1, 2, 0), caption="Prediction")
                    ])

    return total_loss / len(dataloader), total_tar_loss / len(dataloader), wandb_images


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def save_optimizer(optimizer, filename):
    torch.save(optimizer.state_dict(), filename)


def save_losses(epoch, best_val_loss, training_losses, training_tar_losses, validation_losses, validation_tar_losses, filename):
    torch.save({
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'training_losses': training_losses,
        'training_tar_losses': training_tar_losses,
        'validation_losses': validation_losses,
        'validation_tar_losses': validation_tar_losses,
    }, filename)


def load_model(model, filename):
    model.load_state_dict(torch.load(filename))


def load_optimizer(optimizer, filename):
    optimizer.load_state_dict(torch.load(filename))


def load_losses(filename):
    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    training_losses = checkpoint.get('training_losses', [])
    training_tar_losses = checkpoint.get('training_tar_losses', [])
    validation_losses = checkpoint.get('validation_losses', [])
    validation_tar_losses = checkpoint.get('validation_tar_losses', [])
    return epoch, best_val_loss, training_losses, training_tar_losses, validation_losses, validation_tar_losses


def main():
    wandb.init(project=CONFIG['project_name'], config=CONFIG)
    print("Config:", CONFIG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = setup_data(CONFIG['data_dir'])

    model = create_model(CONFIG['model_name']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0)

    start_epoch = 0
    best_val_loss = float('inf')
    training_losses = []
    training_tar_losses = []
    validation_losses = []
    validation_tar_losses = []

    if CONFIG['resume_training']:
        model_pattern = re.compile(r'model_epoch_(\d+)\.pth')
        optimizer_pattern = re.compile(r'optimizer_epoch_(\d+)\.pth')
        loss_pattern = re.compile(r'losses_epoch_(\d+)\.pth')

        model_files = []
        optimizer_files = []
        loss_files = []

        for file_name in os.listdir(os.path.join(CONFIG['model_dir'], CONFIG['model_name'])):
            model_match = model_pattern.match(file_name)
            optimizer_match = optimizer_pattern.match(file_name)
            loss_match = loss_pattern.match(file_name)

            if model_match:
                epoch = int(model_match.group(1))
                model_files.append((epoch, file_name))
            if optimizer_match:
                epoch = int(optimizer_match.group(1))
                optimizer_files.append((epoch, file_name))
            if loss_match:
                epoch = int(loss_match.group(1))
                loss_files.append((epoch, file_name))

        if not model_files or not optimizer_files or not loss_files:
            print("No files found.")
            return

        latest_epoch = max(model_files, key=lambda x: x[0])[0]

        latest_model_file = os.path.join(CONFIG['model_dir'], CONFIG['model_name'], f'model_epoch_{latest_epoch}.pth')
        latest_optimizer_file = os.path.join(CONFIG['model_dir'], CONFIG['model_name'], f'optimizer_epoch_{latest_epoch}.pth')
        latest_loss_file = os.path.join(CONFIG['model_dir'], CONFIG['model_name'], f'losses_epoch_{latest_epoch}.pth')
        
        if os.path.exists(latest_model_file) and os.path.exists(latest_optimizer_file) and os.path.exists(latest_loss_file):
            load_model(model, latest_model_file)
            load_optimizer(optimizer, latest_optimizer_file)
            start_epoch, best_val_loss, training_losses, training_tar_losses, validation_losses, validation_tar_losses = load_losses(latest_loss_file)
            print(f"Resuming training from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")

    os.makedirs(os.path.join(CONFIG['model_dir'], CONFIG['model_name']), exist_ok=True)

    for epoch in range(start_epoch, CONFIG['epochs']):
        train_loss, train_tar_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_tar_loss, wandb_images = validate(model, val_loader, device)

        # Append losses to the list
        training_losses.append(train_loss)
        training_tar_losses.append(train_tar_loss)
        validation_losses.append(val_loss)
        validation_tar_losses.append(val_tar_loss)

        # Prepare log dictionary
        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_tar_loss": train_tar_loss,
            "val_loss": val_loss,
            "val_tar_loss": val_tar_loss,
        }

        # Add images to log dictionary
        for idx in range(len(wandb_images)):
            log_dict[f"predictions_{idx+1}"] = wandb_images[idx]

        # Log everything at once
        wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, os.path.join(CONFIG['model_dir'], CONFIG['model_name'], 'best_model.pth'))
            save_optimizer(optimizer, os.path.join(CONFIG['model_dir'], CONFIG['model_name'], 'best_optimizer.pth'))
            save_losses(epoch, best_val_loss, training_losses, training_tar_losses, validation_losses, validation_tar_losses,
                        os.path.join(CONFIG['model_dir'], CONFIG['model_name'], 'best_losses.pth'))

        if (epoch + 1) % CONFIG['save_freq'] == 0:
            save_model(model, os.path.join(CONFIG['model_dir'], CONFIG['model_name'], f'model_epoch_{epoch+1}.pth'))
            save_optimizer(optimizer, os.path.join(CONFIG['model_dir'], CONFIG['model_name'], f'optimizer_epoch_{epoch+1}.pth'))
            save_losses(epoch, best_val_loss, training_losses, training_tar_losses, validation_losses, validation_tar_losses,
                        os.path.join(CONFIG['model_dir'], CONFIG['model_name'], f'losses_epoch_{epoch+1}.pth'))

        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
