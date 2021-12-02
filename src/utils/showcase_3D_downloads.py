import os
import requests

def download_image(url_base_image, dir_image_store):
    # retrieving the image sample
    image_tag = "50.0"
    extensions = [".mhd", ".raw"]
    for extension in extensions:
        url_image_sample = url_base_image + image_tag + extension
        if not os.path.exists(dir_image_store + image_tag + extension):
            file_image_sample = requests.get(url_image_sample)
            open(dir_image_store + image_tag + extension, 'wb').write(file_image_sample.content)
            print(dir_image_store + image_tag + extension + " downloaded successfully")
        else:
            print(dir_image_store + image_tag + extension + " already exists")

def download_models(url_base_model, dir_model_store):

    # Downloading axial, coronal and sagittal models for the example
    model_tag = "ep160_bs1_lr1e-3_3D_model"

    local_file_path = dir_model_store + model_tag + ".pt"
    if not os.path.exists(local_file_path):
        url_model = url_base_model + model_tag + ".pt"
        file_model = requests.get(url_model)
        open(local_file_path, 'wb').write(file_model.content)
        print(local_file_path + " downloaded successfully")
    else:
        print(local_file_path + " already exists")


def check_dir_exist(dir_list):
    for dir_path in dir_list:
        # check if the needed directories already exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f'{dir_path} directory created')

def run_showcase_3D_downloads(model_only=False):

    url_base_image = "https://www.creatis.insa-lyon.fr/~roux/deep_learning_motion_mask_segmention/image_sample/"
    dir_image_store = "data/image_sample/"

    url_base_model = "https://www.creatis.insa-lyon.fr/~roux/deep_learning_motion_mask_segmention/model_weights/"
    dir_model_store = "data/model_weights/"

    check_dir_exist([dir_image_store, dir_model_store])
    if model_only:
        print("skipping download of the demo images")
        download_models(url_base_model, dir_model_store)
    else:
        download_image(url_base_image, dir_image_store)
        download_models(url_base_model, dir_model_store)




if __name__ == '__main__':
    run_showcase_3D_downloads()
