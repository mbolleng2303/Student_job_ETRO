"""
    File to load dataset based on user control from main file
"""

from data.SBMs import SBMsDataset
from data.Xray_1024.X_ray_1024 import X_ray_1024_Dataset
from data.Xray_1024.X_ray_1024_INFERENCE import X_ray_1024_Dataset_inference
from data.Xray_1024.X_ray_1024_test5 import X_ray_1024_Dataset5


def LoadData(DATASET_NAME, inference =False, inf5 = False, edge = False):
    """
        This function is called in the main.py file
        returns:
        ; dataset object
    """
    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS:
        return SBMsDataset(DATASET_NAME)
    if DATASET_NAME == "X_ray_1024":
        if not inference and not inf5:
            return X_ray_1024_Dataset(DATASET_NAME, edge=edge)
        elif inference:
            return X_ray_1024_Dataset_inference(DATASET_NAME)
        elif inf5:
            return X_ray_1024_Dataset5(DATASET_NAME)


