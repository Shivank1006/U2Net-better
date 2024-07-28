import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import gradio as gr
from data_loader import RescaleT, ToTensorLab
from model import U2NET, U2NETP

# Configuration dictionary
CONFIG = {
    "model_name": "u2net",
    "checkpoint_path": "saved_models/u2net/best_model_512.pth",
    "resolution": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

model_dir = CONFIG['checkpoint_path']

if CONFIG['model_name'] == 'u2net':
    print("...load U2NET---173.6 MB")
    net = U2NET(3, 1)
elif CONFIG['model_name'] == 'u2netp':
    print("...load U2NEP---4.7 MB")
    net = U2NETP(3, 1)

if CONFIG['device'] == 'cuda' and torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:
    net.load_state_dict(torch.load(model_dir, map_location=CONFIG['device']))
net.eval()

# Normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def process_image(image: Image.Image):
    # Convert the PIL image to a numpy array
    image_np = np.array(image)

    # Prepare the input sample for transformations
    input_sample = {'imidx': np.array([0]), 'image': image_np, 'label': np.zeros((CONFIG['resolution'], CONFIG['resolution'], 3))}
    
    preprocess = transforms.Compose([
        RescaleT(CONFIG['resolution']),
        ToTensorLab(flag=0)
    ])

    transformed_sample = preprocess(input_sample)
    image_tensor = transformed_sample['image'].unsqueeze(0).type(torch.FloatTensor)

    if CONFIG['device'] == 'cuda':
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(image_tensor)

    pred = d1[:, 0, :, :]
    pred = normPRED(pred)

    pred = pred.squeeze().cpu().data.numpy()
    im_mask = Image.fromarray((pred * 255).astype(np.uint8)).convert('L')
    im_mask = im_mask.resize(image.size, resample=Image.LANCZOS)

    rgba_image = image.convert("RGBA")
    rgba_image.putalpha(im_mask)

    return rgba_image

gr.Interface(
    fn=process_image, 
    inputs=gr.Image(type="pil", image_mode="RGBA"),
    outputs=gr.Image(type="pil", image_mode="RGBA"),
    title="U2Net Image Segmentation"
).launch(share=True)
