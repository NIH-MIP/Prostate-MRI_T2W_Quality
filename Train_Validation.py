import pandas as pd
import torch
import numpy as np
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
df_manip = pd.read_csv(data)



###############################################
#### LOAD TRAINING/VALIDATION/TESTING DATA ####
###############################################
# create separate pandas dataframes for each data partition
Train_df = df_manip[df_manip.TTV_Split == 'Train']
Validation_df = df_manip[df_manip.TTV_Split == 'Validation']

# create a dictionary 
Train_T2_dict = [{"image": image_name, "label": label} for image_name, label in zip(Train_df.T2_nifti_file,Train_df.T2_GT)]
Validation_T2_dict = [{"image": image_name, "label": label} for image_name, label in zip(Validation_df.T2_nifti_file, Validation_df.T2_GT)]



############################
# Augmentation/Transforms #
###########################
spatial_size = (512,512,32)
roi_size = (320,320,16)
spacing = (.25,.25,-1)
# Define transforms
train_transforms = Compose([# preprocessing
                            LoadImaged(keys=['image']),
                            AddChanneld(keys=['image']),
                            Spacingd(keys = ['image'], pixdim = spacing),
                            NormalizeIntensityd(keys=['image'],nonzero=True),
                            CenterSpatialCropd(keys=['image'], roi_size = roi_size),
                            Resized(keys=['image'], spatial_size = (-1,-1,32)),
                            # augmentations
                            RandStdShiftIntensityd(keys=['image'],prob=0.5,factors=(2,5)),
                            RandHistogramShiftd(keys=['image'],prob=0.5,num_control_points = (3,5)),
                            RandFlipd(keys=['image'],prob=.5,spatial_axis=0),
                            RandFlipd(keys=['image'],prob=.5,spatial_axis=1),
                            RandFlipd(keys=['image'],prob=.5,spatial_axis=2),
                            RandZoomd(keys=['image'],prob=.5,min_zoom = 1.1, max_zoom = 1.5),
                            RandRotated(keys=['image'],prob=.5,range_x=1, range_y=1, range_z=0),
                           ]) 

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
train_ds = Dataset(Train_T2_dict, transform=train_transforms)
# create a data loader
train_batch = 5
train_loader = DataLoader(train_ds, batch_size=train_batch, num_workers=4,shuffle=True)

# Define image dataset
validation_ds = Dataset(Validation_T2_dict, transform=valtest_transforms)
# create a data 6oader
validation_loader = DataLoader(validation_ds, batch_size=1, num_workers=4,shuffle=True)



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



##########################################
#### START TRAINING/VALIDATION EPOCHS ####
##########################################
epoch_num = 100
train_epoch_num = 1
epoch_loss = 0
best_metric = -100
for epoch in range(epoch_num):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    train_loss = 0
    train_step = 0
    ### use ctr+/ to comment in/out blocks of code
    
    for train_epoch in range(train_epoch_num):
        ##################
        #### TRAINING ####
        ##################
        print('Start Training')
        for batch_data in train_loader:
            ####################################################
            # Import mask, apply sliding window, set stride, reshape
            batch_labels = batch_data['label'].to(device)
            hot_labels = torch.nn.functional.one_hot(torch.as_tensor(batch_labels),num_classes=num_prediction_classes).float()
            batch_images = batch_data['image'].to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = loss_function(outputs, hot_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()        
            
            print(loss,epoch_loss)

        
    ####################
    #### VALIDATION ####
    ####################
    print('Start Validation')
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
        for batch_data in validation_loader:
            ####################################################
            # Import mask, apply sliding window, set stride, reshape
            batch_labels = (batch_data['label']).to(device)
            hot_labels = torch.nn.functional.one_hot(torch.as_tensor(batch_labels),num_classes=num_prediction_classes).float()
            batch_images = batch_data['image'].to(device)

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