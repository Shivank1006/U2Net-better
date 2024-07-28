# U2Net-better

## Introduction
U2Net-better is an improved implementation of the U2Net model, which is used for salient object detection. This version introduces several enhancements to the original codebase, with improvements in validation, code readability, and the integration of Weights and Biases (wandb) for tracking losses and validation images to make it more user-friendly, maintainable, and powerful. Additionally, a Gradio user interface is included for easy inference on single images.

## Features
- Validation: Added validation support to evaluate the model's performance during training.
- Code Readability: Improved the readability and organization of the code for better understanding and maintenance.
- wandb Tracking: Integrated Weights and Biases (wandb) for tracking training and validation losses, as well as visualization of validation images.
- Gradio UI: Included a Gradio user interface to perform inference on single images and visualize the results easily.
- Data Preprocessing: Added a preprocessing script to split the dataset into training and testing sets and apply transformations to the training set.
- Bulk Testing: Included a script to perform bulk testing on images, generating combined output images.

## Installation
To get started with U2Net-better, follow these steps:

1. Clone the repository:
```sh
git clone https://github.com/Shivank1006/U2Net-better.git
cd U2Net-better
```

2. Install the required dependencies:
```sh
pip install -r requirements.txt
```

3. Set up wandb:
    - Create an account on Weights and Biases (https://wandb.ai/).
    - Run `wandb login` and enter your API key.

## File Structure

```
U2Net-better/
├── model/
│   ├── __init__.py
│   ├── u2net.py
│   ├── u2net_refactor.py
├── dataset/
│   ├── imgs/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── masks/
│   │   ├── mask1.png
│   │   ├── mask2.png
│   │   └── ...
├── preprocessed_dataset/
│   ├── train/
│   │   ├── images/
│   │   │   ├── image1.jpg
│   │   │   ├── image1_h.jpg (horizontally flipped)
│   │   │   ├── image1_v.jpg (vertically flipped)
│   │   │   └── ...
│   │   ├── masks/
│   │   │   ├── mask1.png
│   │   │   ├── mask1_h.png (horizontally flipped)
│   │   │   ├── mask1_v.png (vertically flipped)
│   │   │   └── ...
│   ├── test/
│   │   ├── images/
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   ├── masks/
│   │   │   ├── mask2.png
│   │   │   └── ...
├── data_loader.py
├── preprocess_dataset.py
├── train.py
├── test.py
├── gradio_inference.py
├── requirements.txt
└── README.md
```

This structure includes:
- `dataset/`: The original dataset folder containing images and masks.
- `preprocessed_dataset/`: The preprocessed dataset folder containing the training and testing sets with applied transformations.
  - `train/`: Training set folder with subfolders for images and masks, including flipped versions.
  - `test/`: Testing set folder with subfolders for images and masks.
- `preprocess_dataset.py`: Script for preprocessing the dataset.
- `train.py`: Script for training the model.
- `test.py`: Script for bulk testing on images.
- `gradio_inference.py`: Script to run the Gradio UI for single image inference.
- `requirements.txt`: File containing the list of dependencies.
- `README.md`: Project documentation file.

This layout provides a clear and organized view of the project structure, making it easier to navigate and understand.

## Usage
### Data Preprocessing
The preprocessing inputs are currently set within the `preprocess_dataset.py` script itself. This script will:
- Split the dataset into training and testing sets.
- Apply horizontal and vertical flip transformations to the training set images.
- Save the preprocessed data in the specified output directory.

To run the preprocessing script:
```sh
python preprocess_dataset.py
```

### Training
To train the model, simply run:
```sh
python train.py
```
The configuration for training is defined as a dictionary within the code itself.

### Inference
To perform inference on a single image using the Gradio UI, run:
```sh
python gradio_inference.py
```
This will launch a web interface where you can upload an image and see the resulting saliency map.

### Bulk Testing
To perform bulk testing on images, run:
```sh
python test.py
```
This script will generate combined images for each test image, showing the original image, predicted mask, and the image with the mask applied as an alpha channel.

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

### Steps to Contribute
1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature-branch).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
- Original U2Net (https://github.com/xuebinqin/U-2-Net) by Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, and Martin Jagersand.
- Weights and Biases (https://wandb.ai/) for providing the platform to track experiments.
- Gradio (https://gradio.app/) for creating the easy-to-use UI for model inference.

## TODO List
- [x] Validation: Added validation support.
- [x] Code Readability: Improved code readability and organization.
- [x] Wandb Tracking: Integrated wandb for tracking training and validation.
- [x] Gradio UI: Added Gradio interface for inference.
- [x] Data Preprocessing: Added data preprocessing script.
- [x] Bulk Testing: Added bulk testing script.
- [x] Resume Training: Implement functionality to resume training from a checkpoint.
- [ ] Config-based Training and Testing: Introduce a config.yml file for configuring training and testing parameters.
- [ ] Loss calculation and validation based on the number of steps instead of epochs.

Thank you for using U2Net-better! We hope you find it useful for your projects. If you have any questions, feel free to open an issue or contact us.
