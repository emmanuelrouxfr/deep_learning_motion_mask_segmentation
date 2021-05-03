import json
import os
import torch
import SimpleITK as sitk
import numpy as np

from src.model.uuunet import UNet
from src.utils.compute_prediction import compute_prediction
from munch import Munch
from tqdm import tqdm

# GLOBAL VARIABLES
cost_function = "Dice"
optimizer = "adam"
input_img_path = "./data/img_demo.mhd"
n_channels = 1
n_classes = 2
batch_size = 32
input_size = [256, 256, 256]
pre_trained_weights_path = {
    "axial": "/home/penarrubia/ShareCluster/penarrubia/segmentation/stage_master_nicolaspinon/results/2_test_fold1_ep20_bs32_lr1e-3_axial/checkpoint_best.pt",
    "coronal" : "/home/penarrubia/ShareCluster/penarrubia/segmentation/stage_master_nicolaspinon/results/2_test_fold1_ep20_bs32_lr1e-3_coronal/checkpoint_best.pt",
    "sagittal" : "/home/penarrubia/ShareCluster/penarrubia/segmentation/stage_master_nicolaspinon/results/2_test_fold1_ep20_bs32_lr1e-3_sagittal/checkpoint_best.pt"
}
lr = 1e-3


def main():

    # Define parameters
    params = Munch()
    params.cost_function = cost_function
    params.optimizer = optimizer
    params.input_img_path = input_img_path
    params.input_size = input_size
    params.n_channels = n_channels
    params.n_classes = n_classes
    params.batch_size = batch_size
    params.pre_trained_weights_path = pre_trained_weights_path
    params.lr = lr
    params.device = "cpu"
    
    # Extract the experiment tag and create the associated folder
    params.exp_tag = "trained_model_on_showcase_data"
    params.results_folder = os.path.join("./results", params.exp_tag)
    os.makedirs(params.results_folder, exist_ok=True)

    # Load data
    image = sitk.ReadImage(params.input_img_path)
    direction = image.GetDirection()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    image_array = sitk.GetArrayFromImage(image)
    image_tensor = torch.tensor(image_array).float()
    image_tensor = image_tensor.unsqueeze(axis=0).unsqueeze(axis=0)
    image_tensor = image_tensor.to(device=params.device)

    # Create empty volume to sum predictions
    pred_sum = np.zeros((params.input_size))

    # Load UNet models and predict volumes in each direction
    for slicing in tqdm(pre_trained_weights_path.keys(), total=3, desc="Direction prediction"):
        model = UNet(params.n_channels, params.n_classes)
        checkpoint = torch.load(pre_trained_weights_path[slicing], map_location=params.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(params.device)
        model.eval()
        pred_array = compute_prediction(model, slicing, image_tensor)
        pred_sum += pred_array
    
    # Majority vote
    prediction = np.where(pred_sum >= 2, 1, 0).astype(np.int16)
    
    # Saving of the prediction
    prediction_image = sitk.GetImageFromArray(prediction.astype(np.int16))
    prediction_image.SetDirection(direction)
    prediction_image.SetOrigin(origin)
    prediction_image.SetSpacing(spacing)
    sitk.WriteImage(prediction_image, os.path.join(params.results_folder, "pred_demo.mhd"))


if __name__ == '__main__':
    main()

