import torch

def compute_prediction(model, slicing, image_tensor):
    axis_slicing = {"axial":0, "coronal":1, "sagittal":2}
    nb_slices = image_tensor.size()[axis_slicing[slicing]+2]
    device = next(model.parameters()).device
    dim_in_tensor = axis_slicing[slicing] + 2 #counting dimensions for channel and batch size

    for index_slice in range(nb_slices):
        index_slice = torch.tensor([index_slice])
        slice_to_predict = torch.index_select(image_tensor, dim_in_tensor , index_slice.to(device)) # shape slice_to_predict : [1, 1, 1, 256, 256]
        slice_to_predict = torch.squeeze(slice_to_predict, dim=dim_in_tensor) # shape of slice to predict : [1, 1, 256, 256]
        predicted_slice = model(slice_to_predict)
        predicted_slice = predicted_slice.detach()
        predicted_slice = torch.squeeze(predicted_slice)  # remove batch dimension
        predicted_slice = torch.argmax(predicted_slice, axis=0) # retrieve mask 
        predicted_slice = torch.unsqueeze(predicted_slice, dim=axis_slicing[slicing]) # shape for concatenation

        if index_slice == 0:
            pred_array = predicted_slice.clone().detach()
            print(pred_array.shape)
        else:
            pred_array = torch.cat((pred_array, predicted_slice), dim=axis_slicing[slicing])
    
    return pred_array