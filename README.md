# deep_learning_motion_mask_segmentation

![image from web](https://www.creatis.insa-lyon.fr/nextcloud/index.php/s/boGJjsP5tnFSWw2/preview)

This repository is used to demonstrate the method published in Medical Physics : https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.15347  

If you use this repository, please **cite** the associated publication :  


[**Improving motion-mask segmentation in thoracic CT with multiplanar U-nets**](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.15347).  
Authors : Ludmilla Penarrubia, Nicolas Pinon, Emmanuel Roux, Eduardo Enrique Davila Serrano, Jean-Christophe Richard, Maciej Orkisz and David Sarrut.

## This repository has several usages :

 1- Use our trained models on the data we provide, as a proof of concept : (a) multi-planar UNet and (b) 3D-UNet.

 2- Test our trained model on your data, to get the motion mask segmentations on your data.  

 3- Train our model on your data and test it on your data  

## Pre-requisites and installations

* Make sure you have python3 installed  

* Clone this repository on your machine and go in it:  

    `cd deep_learning_motion_mask_segmentation/`  

* Create a virtual environments  

    `python3 -m venv motion_mask_seg`  

* Activate the virtual environment  

    `source motion_mask_seg/bin/activate`  

* Update pip3 repository and install dependencies listed in the requirements.txt  

    `pip3 install --upgrade pip`  
    `python3 -m pip install -r requirements.txt`  

* Install Gatetools for preprocessing (optional)

    `pip3 install gatetools`  



## Case 1 (a) : Use *our* trained model on *our* showcase data

   Run :`python3 trained_model_on_showcase_data.py`  
   Motion mask as .mhd and .raw files will be located in : `results_showcase/trained_model_on_showcase_data`  


   Run :`python3 trained_3D_model_on_showcase_data.py`  
   Motion mask as .mhd and .raw files will be located in : `results_showcase/trained_3D_model_on_showcase_data`  

## Case 2 : Use *our* trained model on *your* data

   Put your image sample (e.g., input_data.mhd or .nii) in the directory `data/`  

   (optional)
   Run the following, line by line :
   ```bash
   gatetools/bin/gt_image_gauss -i data/input_data.mhd -o data/inter_data.mhd --sigma_mm "1.0"
   gatetools/bin/gt_affine_transform -i data/inter_data.mhd -o data/output_data.mhd --force_resample --adaptive --newspacing "2.0" 
   ```  

   Edit the file `infer_motion_masks.py` (l. 62-63) with the path to your data and its size:

   ```python
       params.input_img_path = "./data/PATH_OF_THE_IMAGE_TO_SEGMENT.mhd"
   ```  

   Finally run :
   ```bash
   python3 infer_motion_masks.py
   ```


   Motion mask as .nii files and figures will be located in : `results/inference_on_your_data/`  
   We suggest skipping the preprocessing step only if your data is sampled as isotropic 2mm^3  

## (Advanced) Case 3 : Train and test our model on *your* data

For this use-case, we recommend pluggin in your code the model located in model.py, we do not provide the data management part of the code, as it is really specific to each user.  


## Acknowledments

Thanks to the authors of this repository : https://github.com/milesial/Pytorch-UNet for providing an efficient implementation of UNet.  

Thanks to the authors of this repository : https://github.com/UdonDa/3D-UNet-PyTorch/ for providing an efficient implementation of 3D-UNet.  

This work was performed within the framework of the LABEX PRIMES (ANR-11-LABX-0063) of Université de Lyon, within the program "Investissements d'Avenir"(ANR-11-IDEX-0007) operated by the French National Research Agency (ANR).  

Thanks to Olivier Bernard for getting us started with the project by providing examples of his codes.
