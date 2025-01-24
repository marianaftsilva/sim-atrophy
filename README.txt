*Biomechanical Simulation of Brain Atrophy Using Deep Networks*

Author: Mariana da Silva


HOW TO RUN:

To train the model, run the following inside the /sim-atrophy/ folder:

     'python train.py --atr_path /path_to_atrophies/ --seg_path /path_to_segmentations/'


At inference time, run the following:

     'python test.py --atr_path /path_to_atrophies/ --img_path /path_to_images/ --load_model trained_model_3d.pt'

The results of model will be saved in the 'results' folder as NIFTI files. This includes the deformed images, calculated atrophies and deformation field.


DATA REQUIREMENTS:

The model takes NIFTI images and segmentations as input. Atrophies can be provided as an NIFTI volumetric file with the same dimentions of the images or as list of atrophies per region. For lists, you need to also provide the parcellations of the brain.

The atrophies correpond to local volume changes, such that a = V1/V2. This means that values of 1 correspond to no deformation, values >1 correspond to atrophy and values <1 correpond to expansion.

At training time, you need to provide segmentations, where label 0 corresponds to the background, 1 corresponds to the CSF and labels >1 correspond to the brain tissue. The images do not need to be provided at training time.

At test time, you need to provide the baseline images and atrophy maps.

The code assumes that the atrophies, segmentations and images all have the same filename, e.g. 'SubXXXX.nii.gz', and that the train and test data are already separated in different folders.
