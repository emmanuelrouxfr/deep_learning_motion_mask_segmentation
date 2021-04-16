# deep_learning_motion_mask_segmention

![image from web](https://www.creatis.insa-lyon.fr/nextcloud/index.php/s/boGJjsP5tnFSWw2/preview)



We will illustrate our method with one example (download the image (public from COVID CT-PRED) and model weights -from CREATIS website => to check)

preprocessing => (check Thomas B.) say it can work with classic format (SimpleITK).
dependencies => install gate tools

The merging will be done in a class that has several methods :

      predict()
      predict_axial()
      predict_coronal()
      predict_sagittal()
      merge_predictions()


cite https://github.com/milesial/Pytorch-UNet for the UNet model code

## Run the example

1- make sure you have python3 installed

2- clone this repository on your machine and go in it:

    cd deep_learning_motion_mask_segmention/

3- create a virtual environments

    python3 -m venv motion_mask_seg

4- activate the virtual environment

    source motion_mask_seg/bin/activate

5- update pip3 repository and install dependencies listed in the requirements.txt

    pip3 install --upgrade pip
    python3 -m pip install -r requirements.txt

6- Run the example illustrated above

    **TO DO : add command line for running the example**
    python3 main.py

## Notes
