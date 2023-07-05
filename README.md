# Prostate-MRI_T2W_Quality
code repository for Belue et al

This repository presents a 3D DenseNet121 AI model designed to automate the evaluation of T2W (T2-weighted) prostate MRI image quality. This tool is designed to standardize quality assessments and identify suboptimal examinations, reducing subjectivity in the process. 

The model was trained on a dataset of 1046 patients' MRI scans related to prostate cancer suspicion or follow-up. It assigns quality scores based on various distortion evaluations and provides 3D voxel-level quality heatmaps for detailed analysis.

In essence, this repository provides an essential tool for more consistent, automated, and reliable T2W prostate MRI quality evaluations.

---

## Code Files

The repository contains two main Python scripts:
1. `Train_Validation.py`: This script trains the model using training and validation datasets.
2. `Test.py`: This script evaluates the model on a testing dataset, generating occlusion sensitivity maps for each test image.

## Dependencies

The code uses the following libraries:
- pandas v1.4.3
- torch v1.12.1
- numpy v1.23.1
- monai v1.2.0
- nibabel v4.0.1

## Usage

1. Prepare your datasets and update the CSV file path (`data = 'path\\to\\excel\\with\\paths\\data.csv'`) in both scripts.
2. Update the model output directory (`model_output_dir = "path\\to\\model\\weights\\output\\folder\\"`) in `code1.py`.
3. Update the model weights path (`best_model = "path\\to\\best\\model.pth"`) in both scripts.
4. Update the directory to save the occlusion sensitivity maps (`save_dir = "path\\to\\save\\partial\\occlusion\\maps\\"`) in `code2.py`.
5. Run `Train_Validation.py` to train the model.
6. Run `Test.py` to evaluate the model on the test dataset and generate occlusion sensitivity maps.

## Note

The paths for the CSV files, model weights, and occlusion sensitivity maps are placeholders and should be replaced with your actual file and directory paths before running the code.

Remember that you might need to adjust the model parameters and the data augmentation techniques depending on your specific task and data.

The image preprocessing pipeline used in this code assumes that the input images are in the NIfTI format, which is a common format for medical images. If your images are in a different format, you might need to adjust the preprocessing steps accordingly. 

The code is currently configured to use a CUDA-enabled GPU for model training and testing if one is available. If you do not have a CUDA-enabled GPU, the code will default to using the CPU.
