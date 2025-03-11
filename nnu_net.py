# Install nnU-Net
!pip install nnunet

import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nnunet.inference.predict import predict_from_folder
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset Paths (Modify as needed)
image_path = "/content/liver-tumor-segmentation/images"
label_path = "/content/liver-tumor-segmentation/labels"
output_dir = "/content/nnunet_predictions"

# Convert dataset to nnU-Net format
# (nnU-Net requires a specific folder structure: 'imagesTr', 'labelsTr', etc.)

os.makedirs(output_dir, exist_ok=True)

# Define nnU-Net dataset format
nnunet_dataset_path = "/content/nnunet_dataset/Task003_Liver"
os.makedirs(nnunet_dataset_path, exist_ok=True)

os.makedirs(os.path.join(nnunet_dataset_path, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(nnunet_dataset_path, "labelsTr"), exist_ok=True)

# Convert images to nnU-Net format (nnU-Net expects filenames in a specific format)
for i, img_file in enumerate(sorted(os.listdir(image_path))):
    img = nib.load(os.path.join(image_path, img_file))
    lbl = nib.load(os.path.join(label_path, os.listdir(label_path)[i]))

    img_nnunet_name = f"liver_{i:04d}_0000.nii.gz"
    lbl_nnunet_name = f"liver_{i:04d}.nii.gz"

    nib.save(img, os.path.join(nnunet_dataset_path, "imagesTr", img_nnunet_name))
    nib.save(lbl, os.path.join(nnunet_dataset_path, "labelsTr", lbl_nnunet_name))

print("Dataset converted for nnU-Net.")

# nnU-Net Training (Modify folds and GPU ID as needed)
trainer = nnUNetTrainerV2("3d_fullres", fold=0)
trainer.initialize(training=True)
trainer.run_training()

# Load the trained nnU-Net model for inference
model_folder = trainer.output_folder
model, _ = load_model_and_checkpoint_files(model_folder, folds=[0])

# Run inference on validation images
predict_from_folder(
    model=model,
    input_folder=os.path.join(nnunet_dataset_path, "imagesTr"),
    output_folder=output_dir,
    folds=[0],
    save_npz=False,
    num_threads_preprocessing=1,
    num_threads_nifti_save=1,
    disable_tta=True,
)

print("Inference complete. Predictions saved in:", output_dir)

# Load and visualize a prediction
pred_file = sorted(os.listdir(output_dir))[0]  # Take first prediction
pred_img = nib.load(os.path.join(output_dir, pred_file)).get_fdata()

# Visualize the result
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(pred_img[:, :, pred_img.shape[2] // 2], cmap="gray")
plt.title("Predicted Segmentation (nnU-Net)")

plt.subplot(1, 2, 2)
original_img = nib.load(os.path.join(nnunet_dataset_path, "imagesTr", pred_file.replace("_0000", ""))).get_fdata()
plt.imshow(original_img[:, :, original_img.shape[2] // 2], cmap="gray")
plt.title("Original Image")

plt.show()
