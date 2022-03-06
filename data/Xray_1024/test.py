import pandas as pd
import pandas as pd
import os
import numpy as np
df = pd.read_csv('training_data_features_and_outputs_full.csv',
                 sep=",", header=None,
                 names=['PatientID' ,'Subset' ,'Age' ,'Sex' ,'Covid+' ,'Severity' ,'Covid+ & severe' ,'Covid- & severe'
                        ,'Image_size_x' ,'Image_size_y' ,'Image_size_z' ,'Spacing_x' ,'Spacing_y' ,'Spacing_z','cls_Covid_P_VS_N_feat','cls_Covid_P_VS_N_out','cls_Severe_VS_NSevere_feat_fold0','cls_Severe_VS_NSevere_out_fold0','cls_Severe_VS_NSevere_feat_fold1','cls_Severe_VS_NSevere_out_fold1','cls_Severe_VS_NSevere_feat_fold2','cls_Severe_VS_NSevere_out_fold2','cls_Severe_VS_NSevere_feat_fold3','cls_Severe_VS_NSevere_out_fold3','cls_Severe_VS_NSevere_feat_holdout0','cls_Severe_VS_NSevere_out_holdout0','cls_Severe_VS_NSevere_feat_holdout1','cls_Severe_VS_NSevere_out_holdout1'],skiprows=1)
## How many folds do i have:
## How many folds do i have:
folds = df['Subset'].unique()

idx = np.where((folds != 'holdout0') & (folds != 'holdout1'))
folds = folds[idx]
f= 0
arr = np.zeros([1024, 4, 400])
lab = np.zeros([1, 4, 400])
age_list= np.zeros([4, 400])
data_val=[]
sex_list=np.zeros([4, 400])

for fold in folds:
    ##eval->test data...
    index_val = [k for k, x in enumerate(df['Subset'].values.tolist()) if (x == fold and x != 'holdout0' and x != 'holdout1')]
    data_val = df.iloc[index_val]
    #age = data_val['Age'].values.tolist()
    age= np.array(data_val['Age'].values.tolist())
    sex = np.array(data_val['Sex'].values.tolist())
    feature = np.array(data_val['cls_Severe_VS_NSevere_feat_'+fold].values.tolist())
    labels_val = np.array(data_val['Severity'].to_numpy(dtype=np.int8))
    for i in range(len(feature)):
        arr[:,f,i] = np.array(eval(feature[i]))
        lab[0,f,i] = labels_val[i]
        age_list[f, i] = age[i]/100
        if sex[i]=='F':
            sex_list[f, i] =1
        else :
            sex_list[f, i] =0

    f+=1


arr.tofile('X_ray_1024_feat.csv', sep=',')
lab.tofile('X_ray_1024_label.csv', sep=',')
sex_list.tofile('X_ray_1024_sex.csv', sep=',')









    #files_dict_val = [{'image_ct': feature, 'Age': Age, 'Sex': Sex, 'label': label}
                        #for image_ct, Age, Sex, label in zip(data_val, age, sex, labels_val)]


"""Custom loss functions"""

import torch
from torch.nn import functional as F
from torch.autograd import Variable

def cosine(input1, input2):
    res = input1[0]*input2[0]+input1[1]*input2[1]
    res = res / (((input1[0]**2 +input1[1]**2)**0.5)*(input2[0]**2 +input2[1]**2)**0.5)
    return res

def softmax_mse(input1, input2):
    assert input1.size() == input2.size()
    input_softmax = F.softmax(input1, dim=1)
    target_softmax = F.softmax(input2, dim=1)
    feat = input1.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / feat


def softmax_kl(input1, input2):
    assert input1.size() == input2.size()
    input_log_softmax = F.log_softmax(input1, dim=1)
    target_softmax = F.softmax(input2, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def MSE(input1, input2):
    assert input1.size() == input2.size()
    input1 = F.normalize(input1, dim=-1, p=2)
    input2 = F.normalize(input2, dim=-1, p=2)
    return torch.sum(2 - 2 * (input1 * input2))  ###2 - 2 * (input1 * input2).sum(dim=-1) ###recheck this one


def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    feat = input1.size()[1]
    return torch.sum((input1 - input2) ** 2) / feat


def poly_kernel(input1, d=0.5, alpha=1.0, c=2.0):
    K_XX = torch.mm(input1, input1.t()) + c
    return K_XX.pow(d)


def smi(input1, input2):
    K_X = poly_kernel(input1)
    K_Y = poly_kernel(input2)
    n = K_X.size(0)
    phi = K_X * K_Y
    hh = torch.mean(phi, 1)
    Hh = K_X.mm(K_X.t()) * K_Y.mm(K_Y.t()) / n ** 2 + torch.eye(n)
    alphah = torch.matmul(torch.inverse(Hh), hh)

    smi = 0.5 * torch.dot(alphah, hh) - 0.5
    return smi  # , alphah

e = np.zeros([f, 400, 400])

tresh = 0.75
res = []
for k in range (4) :
    for i in range(400) :
        for j in range(400):
            a =torch.tensor(np.array([[sex_list[k, i]], [age_list[k, i]]]))
            b =torch.tensor(np.array([[sex_list[k, j]], [age_list[k, j]]]))
            e[k, i, j] = cosine(a,b)
            res.append(e[k, i, j])

            if abs(e[k, i, j]) < tresh :
                e[k, i, j]=0
            else :
                e[k, i, j]=1



import matplotlib.pyplot as plt

#e = np.reshape(e,[1,-1])
plt.hist(res, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
e.tofile('X_ray_1024_edge.csv', sep=',')
