import torch
import numpy as np


def compute_prediction(model, slicing, image_tensor):
    axis_slicing = {"axial":0, "coronal":1, "sagittal":2}
    nb_slices = image_tensor.size()[axis_slicing[slicing]+2]
    
    for index_slice in range(nb_slices):
        image_slice = np.take(image_tensor, index_slice, axis=axis_slicing[slicing]+2)
        predicted_slice = model(image_slice)
        predicted_slice = predicted_slice.detach().numpy()  
        predicted_slice = np.squeeze(predicted_slice)  # retrieve batch dimension
        predicted_slice = np.argmax(predicted_slice, axis=0)  
        predicted_slice = np.expand_dims(predicted_slice, axis=axis_slicing[slicing])

        if index_slice == 0:
            pred_array = np.copy(predicted_slice)
        else:
            pred_array = np.append(pred_array, predicted_slice, axis=axis_slicing[slicing])
    
    return pred_array