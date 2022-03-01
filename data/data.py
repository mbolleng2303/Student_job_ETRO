"""
    File to load dataset based on user control from main file
"""

from data.SBMs import SBMsDataset
from data.Xray_1024.X_ray_1024 import X_ray_1024_Dataset


def LoadData(DATASET_NAME):
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
        return X_ray_1024_Dataset(DATASET_NAME)

