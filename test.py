import os
import glob
from tqdm import tqdm
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP

def get_config() -> Dict[str, Any]:
    """Create and return the configuration dictionary."""
    return {
        "model_name": "u2net",  # or "u2netp"
        "image_dir": os.path.join(os.getcwd(), 'preprocessed_dataset', 'test', 'images'),
        "output_dir": os.path.join(os.getcwd(), 'preprocessed_dataset', 'output_combined'),
        "model_path": 'saved_models/u2net/best_model_512.pth',
        "resolution": 512,
        "batch_size": 1,
        "num_workers": 4,
        "use_gpu": torch.cuda.is_available(),
    }

def normalize_prediction(prediction: torch.Tensor) -> torch.Tensor:
    """Normalize the predicted SOD probability map."""
    return (prediction - prediction.min()) / (prediction.max() - prediction.min())

def save_combined_image(image_path: str, prediction: torch.Tensor, output_dir: str) -> None:
    """Save a combined image with original, prediction mask, and RGBA result."""
    prediction = prediction.squeeze().cpu().numpy()

    original_image = Image.open(image_path).convert("RGB")
    mask = Image.fromarray((prediction * 255).astype(np.uint8)).convert('L')
    mask = mask.resize(original_image.size, resample=Image.LANCZOS)
    
    rgba_image = original_image.copy()
    rgba_image.putalpha(mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Prediction Mask')
    axes[1].axis('off')
    
    axes[2].imshow(rgba_image)
    axes[2].set_title('Image with Alpha Mask')
    axes[2].axis('off')

    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def load_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Load the U2NET or U2NETP model."""
    model_name = config['model_name']
    model_path = config['model_path']

    if model_name == 'u2net':
        model = U2NET(3, 1)
    elif model_name == 'u2netp':
        model = U2NETP(3, 1)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    device = torch.device("cuda" if config['use_gpu'] else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

def get_dataloader(image_paths: List[str], config: Dict[str, Any]) -> DataLoader:
    """Create a DataLoader for the test images."""
    dataset = SalObjDataset(
        img_name_list=image_paths,
        lbl_name_list=[],
        transform=transforms.Compose([
            RescaleT(config['resolution']),
            ToTensorLab(flag=0)
        ])
    )
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

def process_images(model: torch.nn.Module, dataloader: DataLoader, image_paths: List[str], config: Dict[str, Any]) -> None:
    """Process images through the model and save results."""
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Processing Images")):
            image = data['image'].to(device).float()
            d1, *_ = model(image)
            pred = normalize_prediction(d1[:, 0, :, :])
            save_combined_image(image_paths[i], pred, config['output_dir'])

def main():
    config = get_config()
    os.makedirs(config['output_dir'], exist_ok=True)

    image_paths = glob.glob(os.path.join(config['image_dir'], '*'))
    print(f"Found {len(image_paths)} images for processing.")

    model = load_model(config)
    dataloader = get_dataloader(image_paths, config)

    process_images(model, dataloader, image_paths, config)

if __name__ == "__main__":
    main()