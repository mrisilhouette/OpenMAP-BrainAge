from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import torchio as tio

import numpy as np
import pandas as pd

import os

class ADNIDataset(Dataset):
    def __init__(self, data_path, target_file, parcellation=False, volume=False, transform=None, normalize=True, mmse_score=False, moca_score=False, img_size=[128, 128, 30], original=False):
        """
        parcellation: True: output the parcellation mapping. Flase: output the original intensity.
        volume: volume vector.
        normalize: whether to normalize the parcellation maapping.
        img_size: img_size for each view.
        original: True: output the whole 3d chunk after the transformation.
        """
        
        self.target_file = os.path.join(data_path, target_file)
        self.data_path = data_path
        self.data = pd.read_csv(self.target_file)
        self.transform = transform
        
        self.parcellation = parcellation
 
        self.volume = volume
        
        self.img_size = img_size
        
        self.normalize = normalize
        self.mmse_score = mmse_score
        self.moca_score = moca_score

        self.original = original
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_path, "registered_data", self.data.iloc[idx]["path_full"][1:])
        mapping_path = os.path.join(self.data_path, "registered_data", self.data.iloc[idx]["path_rigid_parcellation_mapping"][1:])
        volume_path = os.path.join(self.data_path, "registered_data", self.data.iloc[idx]["path_rigid_parcellation_volume"][1:])
        
        df_volume = pd.read_csv(volume_path)
        volume_vec = df_volume.iloc[0].values.flatten()[1:]
        volume_vec = np.float32(volume_vec) # to make it compatible with OASIS3

        label = self.data.iloc[idx]["age"]
        
        subject = tio.Subject(
                    image=tio.ScalarImage(image_path),
                    mapping=tio.LabelMap(mapping_path),
                    )
        
        if self.transform is not None:
            subject_tf = self.transform(subject)
        else:
            subject_tf = subject
        
        crop_transform_sagittal = tio.CropOrPad([5, self.img_size, self.img_size] if type(self.img_size) is int   # Left-right, back - front, bottom - top
                                  else [list(self.img_size)[2], list(self.img_size)[0], list(self.img_size)[1]]
                    )
        crop_transform_coronal = tio.CropOrPad([self.img_size, 5,self.img_size] if type(self.img_size) is int   # Left-right, back - front, bottom - top
                                  else [list(self.img_size)[0], list(self.img_size)[2], list(self.img_size)[1]]
                    )
        crop_transform_axial = tio.CropOrPad([self.img_size,self.img_size, 5] if type(self.img_size) is int   # Left-right, back - front, bottom - top
                                  else [list(self.img_size)[0], list(self.img_size)[1], list(self.img_size)[2]]
                    )
        
        
        # original image in shape of sagittal x coronal x axial
        subject_tf_sagittal = crop_transform_sagittal(subject_tf)
        subject_tf_coronal = crop_transform_coronal(subject_tf)
        subject_tf_axial = crop_transform_axial(subject_tf)
        
        if self.parcellation:
            scale = 1
            if self.normalize:
                scale = 280
            # we need to shuffle the depth axis to the last dim:
            sagittal_data = subject_tf_sagittal['mapping'].data / scale
            sagittal_data = sagittal_data.permute([0,2,3,1])
            coronal_data = subject_tf_coronal['mapping'].data / scale
            coronal_data = coronal_data.permute([0,1,3,2])
            axial_data = subject_tf_axial['mapping'].data / scale
            axial_data = axial_data.permute([0,1,2,3])
        else:
            # we need to shuffle the depth axis to the last dim:
            sagittal_data = subject_tf_sagittal['image'].data
            sagittal_data = sagittal_data.permute([0,2,3,1])
            coronal_data = subject_tf_coronal['image'].data
            coronal_data = coronal_data.permute([0,1,3,2])
            axial_data = subject_tf_axial['image'].data
            axial_data = axial_data.permute([0,1,2,3])
        """    
        img_data = subject_tf['image'].data # torch tensor
        #mapping_data = subject_tf['mapping'].data # torch tensor
        
        # tio requires shape of channels, x, y, z # z is the depth dimension
        img_data = img_data.permute([0, 1, 3, 2])
        #mapping_data = mapping_data.permute([0, 1, 3, 2])
        # we manually interpolate the scale of mapping into 0 - 1
        #mapping_data = mapping_data / 280 #since there are manximum of 280 label types
       """
        
        outputs = []
        outputs.append(sagittal_data)
        outputs.append(coronal_data)
        outputs.append(axial_data)
        
        if self.original:
            if self.parcellation:
                outputs.append(subject_tf['mapping'].data)
                outputs.append(subject_tf['mapping'].affine)
            else:
                outputs.append(subject_tf['image'].data)
                outputs.append(subject_tf['image'].affine)
        
        if self.volume:
            outputs.append(volume_vec)
            
        if self.mmse_score:
            mmse_score = self.data.iloc[idx]["mmse"]
            outputs.append(mmse_score)
        if self.moca_score:
            moca_score = self.data.iloc[idx]["moca"]
            outputs.append(moca_score)

        outputs.append(label)
        
        return outputs

