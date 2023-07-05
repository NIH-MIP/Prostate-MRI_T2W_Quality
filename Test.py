import pandas as pd
import torch
import numpy as np
import nibabel as nib
import monai
from monai.data import DataLoader, Dataset
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    Spacingd,
    NormalizeIntensityd,
    CenterSpatialCropd,
    Resized,
    RandStdShiftIntensityd,
    RandHistogramShiftd,
    RandFlipd,
    RandZoomd,
    RandRotated,
)

def save_preprocessed_filename(Image,save_dir):
    save_transforms = Compose([SaveImaged(keys=['image'], meta_keys = 'image_meta_dict', output_dir = save_dir, output_postfix = 'preprocessed', resample=False)]) 
    save_transforms(Image)



###################################
#### Load image paths from csv ####
###################################
### column definitions ###
### each row represents one patient scan ###
# T2_nifti_file = full path to nifti
# T2_GT = ternary ground truth (0 = non-diagnostic, 1 = equivocal, 2 = diagnostic)
# TTV_Split = train-test-validation data splits ("Train","Validation","Test")

data = 'path\\to\\excel\\with\\paths\\data.csv'
model_output_dir = "path\\to\\model\\weights\\output\\folder\\"
save_dir "path\\to\\save\\partial\\occlusion\\maps\\"
df_manip = pd.read_csv(data)



###############################################
#### LOAD TRAINING/VALIDATION/TESTING DATA ####
###############################################
# create separate pandas dataframes for each data partition
Test_df = df_manip[df_manip.TTV_Split == 'Test']

# create a dictionary 
Test_T2_dict = [{"image": image_name, "label": label} for image_name, label in zip(Test_df.T2_nifti_file,Test_df.T2_GT)]



############################
# Augmentation/Transforms #
###########################
spatial_size = (512,512,32)
roi_size = (320,320,16)
spacing = (.25,.25,-1)
# Define transforms
valtest_transforms = Compose([# preprocessing
                            LoadImaged(keys=['image']),
                            AddChanneld(keys=['image']),
                            Spacingd(keys = ['image'], pixdim = spacing),  
                            NormalizeIntensityd(keys=['image'],nonzero=True),
                            CenterSpatialCropd(keys=['image'], roi_size = roi_size),
                            Resized(keys=['image'], spatial_size = (-1,-1,32)),
                           ])



#########################################
# Training/Validation/Test Data Loaders #
#########################################

# Define image dataset
Test_ds = Dataset(Test_T2_dict, transform=valtest_transforms)
# create a data 6oader
Test_loader = DataLoader(Test_ds, batch_size=1, num_workers=4,shuffle=False)



###################################
#### MODEL CREATION/PARAMETERS ####
###################################

#### DEFINE MODEL/TRAINING PARAMETERS ####
LR = 3e-4
dropout_prob = 0
num_prediction_classes = 3

##### DEFINE MODEL #####
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=num_prediction_classes,progress=True,dropout_prob=dropout_prob).to(device)
weight = torch.tensor([5,2.5,1]).type(torch.cuda.FloatTensor).to(device)
loss_function = torch.nn.CrossEntropyLoss(weight=weight) #weight=weight
optimizer = torch.optim.Adam(model.parameters(), LR)



############################
#### Load model weights ####
############################
best_model = "path\\to\\best\\model.pth"
tensor_dict = torch.load(best_model,map_location=device)
model.load_state_dict(tensor_dict)
model.to(device)

        
#################
#### Testing ####
#################
print('Start Testing')
y_pred = []
y_test = []
metric_values = []
num_correct = 0
metric_count = 0
QD_count = 0
QD_detection = 0
nonQD_count = 0
nonQD_detection = 0
model.eval()

with torch.no_grad():
    for batch_data in Test_loader:
        ####################################################
        # Import mask, apply sliding window, set stride, reshape
        batch_labels = (batch_data['label']).to(device)
        hot_labels = torch.nn.functional.one_hot(torch.as_tensor(batch_labels),num_classes=num_prediction_classes).float()
        batch_images = batch_data['image'].to(device)
        original_spacing = np.array(batch_data['image_meta_dict']['pixdim'][0].cpu())
        new_affine = np.array(batch_data['image_meta_dict']['affine'][0].cpu())        
        
        filename = os.path.splitext(batch_data['image_meta_dict']['filename_or_obj'][0].split("\\")[-1]) # filename of input nifti
        save_data = [save_preprocessed_filename(Image,save_dir) for Image in decollate_batch(batch_data)]   # saves preprocessed images
        
        #####################################
        ### Get occlusion sensitivity map ###
        #####################################
        ######################################################################################################################
        #Get the occlusion sensitivity map
        occ_sens = monai.visualize.OcclusionSensitivity(nn_module=model, mask_size=[32,32,2], n_batch=1, stride=[32,32,2])
        # Only get a single slice to save time.
        # For the other dimensions (channel, width, height), use
        # -1 to use 0 and img.shape[x]-1 for min and max, respectively
        depth_slice = batch_images.shape[-1] // 2
        #occ_sens_b_box = [-1,-1,-1, -1, -1, -1,depth_slice-1, depth_slice]  
        occ_sens_b_box = [-1,-1,-1, -1, -1, -1,-1, -1]  
        occ_result, _ = occ_sens(x=batch_images, b_box=occ_sens_b_box)
        occ_result = occ_result[0, batch_labels.argmax().item()][None]    

        occ_result_C1 = np.array(occ_result[0,:,:,:,0].cpu())
        occ_result_C2 = np.array(occ_result[0,:,:,:,1].cpu())
        occ_result_C3 = np.array(occ_result[0,:,:,:,2].cpu())

        nifti_img_C1 = nib.Nifti1Image(occ_result_C1, new_affine)
        nifti_img_C1.pixdim = new_spacing
        nib.save(nifti_img_C1,save_dir + filename +'occMap_C1.nii') # non-diagnostic occlusion map save

        nifti_img_C2 = nib.Nifti1Image(occ_result_C2, new_affine)
        nifti_img_C2.pixdim = new_spacing
        nib.save(nifti_img_C2,save_dir + filename + '_occMap_C2.nii') # equivocal occlusion map save

        nifti_img_C3 = nib.Nifti1Image(occ_result_C3, new_affine)
        nifti_img_C3.pixdim = new_spacing
        nib.save(nifti_img_C3,save_dir + filename + '_occMap_C3.nii') # diagnostic occlusion map save
        ######################################################################################################################
        


        outputs = model(batch_images)
        preds_integer = outputs.argmax(axis=1)
        y_pred.append(preds_integer.item())
        GT_integer = batch_labels.item()
        y_test.append(GT_integer)

        value = torch.eq(preds_integer, GT_integer)
        metric_count += len(value)
        num_correct += value.sum().item()

        # calcualting binary accuracy based on quality distortion detection
        if (GT_binary == 0) or (GT_binary == 1):
            QD_count += 1

            if preds_binary <= 1:
                QD_detection += 1

        elif GT_binary == 2:
            nonQD_count += 1

            if preds_binary == GT_binary:
                nonQD_detection += 1


    metric = num_correct/metric_count # ternary accuracy
    binary_accuracy = (QD_detection + nonQD_detection)/(QD_count + nonQD_count) # binary accuracy
    avg_metric = (metric + binary_accuracy)/2 # combined accuracy

    binary_sens = QD_detection/QD_count
    binary_spec = nonQD_detection/nonQD_count
    metric_values.append(metric)

    print(metric,binary_accuracy,avg_metric)        


    if (avg_metric>best_metric): #(metric > best_metric) and 
        best_metric = avg_metric
        print('New Metric = ' + str(best_metric))
        model_name = "model1"
        torch.save(model.state_dict(), model_output_dir+model_name+".pth")  