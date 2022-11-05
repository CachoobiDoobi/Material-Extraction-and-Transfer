# Material Transfer
 This is the repository for the project of the CSE4365 Applied Image Processing Final project, baed on the work of  [Benitez-Garcia, Takahashi, and Yanai](https://www.researchgate.net/publication/363934806_Material_Translation_Based_on_Neural_Style_Transfer_with_Ideal_Style_Image_Retrieval)

 - Input: source image, boolean mask (optional), folder of texture(s)
 - Output: material transfered image

## Overview

1. A suitable material image is found based on the selected material type and content image (material_selection.py lines 22-57) 
2. The mask is computed using Mask R-CNN or read from a file (mask.py lines 7-35)
3. Style transfer is peformed using the material image on the target image, based on the work of [Gatys 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) (material_transfer.py lines 242-257, entire file is important)
4. New image and old image are combined using the mask (mask.py lines 38-61)
    
## Installation
- Install Conda: 
    https://conda.io/projects/conda/en/latest/user-guide/install/windows.html
- Open the Anaconda promptand navigate to the project folder
- Create a Conda environment with the given environment.yml: 
    conda env create -f environment.yml 
- Activate environment:
    conda activate assignment3

## Running the code
1. Specify file paths in main.py
2. Run ``python3 main.py`` in the console

