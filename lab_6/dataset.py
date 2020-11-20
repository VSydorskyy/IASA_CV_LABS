from os.path import join as pjoin
from typing import Optional, Tuple

import cv2
import torch
import pandas as pd

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        path_col: str,
        target_col: str, 
        transforms: Optional[object] = None, 
        augmentations: Optional[object] = None,
    ):
        self.image_names = df[path_col].tolist()
        self.targets = df[target_col].tolist()
        
        self.transforms = transforms
        self.augmentations = augmentations

    def __getitem__(self, idx: int):
        
        image_name = self.image_names[idx]
        target = self.targets[idx]

        img = cv2.imread(image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.augmentations is not None:
            img = self.augmentations(image=img)['image']
            
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, target

    def __len__(self):
        return len(self.image_names)
