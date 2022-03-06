
import pandas as pd
import os
import numpy as np


df = pd.read_csv('training_data_features_and_outputs_full.csv',
                 sep=",", header=None,
                 names=['PatientID' ,'Subset' ,'Age' ,'Sex' ,'Covid+' ,'Severity' ,'Covid+ & severe' ,'Covid- & severe'
                        ,'Image_size_x' ,'Image_size_y' ,'Image_size_z' ,'Spacing_x' ,'Spacing_y' ,'Spacing_z','cls_Covid_P_VS_N_feat','cls_Covid_P_VS_N_out','cls_Severe_VS_NSevere_feat_fold0','cls_Severe_VS_NSevere_out_fold0','cls_Severe_VS_NSevere_feat_fold1','cls_Severe_VS_NSevere_out_fold1','cls_Severe_VS_NSevere_feat_fold2','cls_Severe_VS_NSevere_out_fold2','cls_Severe_VS_NSevere_feat_fold3','cls_Severe_VS_NSevere_out_fold3','cls_Severe_VS_NSevere_feat_holdout0','cls_Severe_VS_NSevere_out_holdout0','cls_Severe_VS_NSevere_feat_holdout1','cls_Severe_VS_NSevere_out_holdout1'],skiprows=1)
## How many folds do i have:

folds = df['Subset'].unique()

idx =np.where((folds != 'holdout0') & (folds != 'holdout1'))
folds =folds[idx]

##divide in train and test
for fold in range(4):
    index_val = [k for k, x in enumerate(df['Subset'].values.tolist()) if (x != 'holdout0' and x != 'holdout1')]
    data_train = df.iloc[index_val]
    age = np.array(data_train['Age'].values.tolist())
    sex = np.array(data_train['Sex'].values.tolist())
    feature = data_train['cls_Severe_VS_NSevere_feat_fold'+k].values.tolist()
    labels_val= data_train['Severity'].to_numpy(dtype=np.int8)


    files_dict_val = [{'image_ct': image_ct, 'Age': Age, 'Sex': Sex, 'label': label}
                      for image_ct, Age, Sex, label in zip(feature , age, sex, labels_val)]
    arr = np.zeros((1024,4,400))
    a = data_train['cls_Severe_VS_NSevere_feat_fold0'].values.tolist()
    arr = []

    for item in files_dict_val :
        fold = item['image_ct']['Subset'].values
        feat = item['image_ct']





k+=1

arr = np.zeros([1024, 4, 1200])
for j in range(400):
    for feature in data_train['cls_Severe_VS_NSevere_feat_fold' + str(k)].values.tolist():
        if feature != 'nan':
            try:
                a = eval(feature)
                arr[:, k, j] = np.array(a)
            except TypeError:
                continue


