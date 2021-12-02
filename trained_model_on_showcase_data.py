import json
import os
import torch
import SimpleITK as sitk
import numpy as np
from munch import Munch
from tqdm import tqdm

from src.model.unet import UNet
from src.utils.compute_prediction import compute_prediction
from src.utils.showcase_downloads import run_showcase_downloads



def main(params):

    params.results_folder = os.path.join("./results/showcase", params.exp_tag)
    os.makedirs(params.results_folder, exist_ok=True)

    # download showcase image and models
    run_showcase_downloads()

    # Load data
    image = sitk.ReadImage(params.input_img_path)
    direction = image.GetDirection()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    image_array = sitk.GetArrayFromImage(image)
    image_tensor = torch.tensor(image_array).float()
    image_tensor = image_tensor.unsqueeze(axis=0).unsqueeze(axis=0)
    image_tensor = image_tensor.to(device=params.device)
    params.input_size = list(image.GetSize())
    params.input_size[0] = params.input_size[2]
    params.input_size[2] = params.input_size[1]

    # Create empty volume to sum predictions
    pred_sum = torch.zeros((params.input_size)).to(params.device)

    # Load UNet models and predict volumes in each direction
    for slicing in tqdm(params.pre_trained_weights_path.keys(), total=3, desc="Direction prediction"):
        model = UNet(params.n_channels, params.n_classes)
        checkpoint = torch.load(params.pre_trained_weights_path[slicing], map_location=params.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(params.device)
        model.eval()
        pred_array = compute_prediction(model, slicing, image_tensor)
        pred_sum += pred_array

    # Majority vote
    prediction = torch.where(pred_sum >= 2, 1, 0)

    # Saving of the prediction
    prediction = prediction.detach().cpu().numpy().astype(np.int16)
    prediction_image = sitk.GetImageFromArray(prediction)
    prediction_image.SetDirection(direction)
    prediction_image.SetOrigin(origin)
    prediction_image.SetSpacing(spacing)
    sitk.WriteImage(prediction_image, os.path.join(params.results_folder, "pred_demo.mhd"))


if __name__ == '__main__':

    # Define parameters
    params = Munch()
    params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    params.input_img_path = "./data/image_sample/50.0.mhd"
    params.n_channels = 1
    params.n_classes = 2
    params.pre_trained_weights_path = {
        "axial": "./data/model_weights/fold1_ep20_bs32_lr1e-3_axial.pt",
        "coronal" : "./data/model_weights/fold1_ep20_bs32_lr1e-3_coronal.pt",
        "sagittal" : "./data/model_weights/fold1_ep20_bs32_lr1e-3_sagittal.pt",
    }

    # Extract the experiment tag and create the associated folder
    params.exp_tag = "trained_model_on_showcase_data"

    main(params)
