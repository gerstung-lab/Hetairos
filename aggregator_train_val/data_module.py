import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import ast
import pandas as pd
from .utils import *
from dataclasses import dataclass
from typing import Union


# Data class to store information about each sample in the dataset
@dataclass
class Sample:
    data_path: str  # Path to the data file (.pt file containing tensor data)
    label: Union[int, torch.Tensor]  # Label for the sample, can be an integer or tensor
    slide_id: str  
    age: Union[int, float] 
    location: str  


# Custom dataset class for handling the dataset of slide images
class LocalDataset(Dataset):
    """
    A custom PyTorch Dataset for managing slide images, including data loading, augmentation, 
    and label generation for training and testing.
    
    Attributes:
        data_dir (str): Directory where data files are stored.
        split (list): List of slide IDs to be included in the dataset.
        labels (DataFrame): DataFrame containing label information for each slide.
        mapping (dict): Mapping from label names to numerical labels.
        mode (str): Either 'train' or 'test', to indicate the current mode.
        augmentation (bool): Whether to apply data augmentation.
        soft_labels (bool): Whether to use soft labels.
        loc_dict (dict): Dictionary containing location data for each slide.
        dim_age_embed (int): Dimensionality of age embedding.
    """
    def __init__(self, data_dir, split, labels, mapping, mode, loc_dict, soft_labels=False, augmentation=False, **kwargs):
        self.data_dir = data_dir
        self.data_split = split
        self.labels = labels.set_index('slide')  # Set slide as the index for easy lookup
        self.labels.index = self.labels.index.astype(str)
        self.mapping = mapping 
        self.mode = mode  # Mode can be 'train' or 'test'
        self.augmentation = augmentation  
        self.soft_labels = soft_labels 
        self.loc_dict = loc_dict  
        self.dim_age_embed = kwargs['dim_age_embed']  

        existing_files = set(os.listdir(self.data_dir))
        existing_files = [os.path.splitext(x)[0] for x in existing_files]
        self.drop_prob = kwargs['age_loc_drop_prob']  # Probability to drop age/location information
        self.aug_prob = kwargs['aug_prob']  # Probability of applying augmentation during training
        self.cls_count = kwargs['cls_count']  # Dimension of the output classification vector (could be larger than the number of classes)

        self.data = []  # List to store valid samples
        for slide in self.data_split:
            slide = str(slide)
            # Ensure slide exists and label information is available
            if slide in existing_files and slide in self.labels.index:
                if pd.notna(self.labels.loc[slide]['family']):
                    if self.soft_labels:
                        label = torch.tensor(ast.literal_eval(self.labels.loc[slide]['prob_vector']))  # Soft label as probability vector
                    else:
                        label = int(self.mapping[self.labels.loc[slide]['family']])  # Convert family label to integer label
                    
                    # Append valid sample to the dataset
                    self.data.append(Sample(data_path=os.path.join(self.data_dir, slide + '.pt'), 
                                            label=label, slide_id=slide,
                                            age=self.labels.loc[slide]['age'], 
                                            location=self.labels.loc[slide]['location']))
                else:
                    print(f"Slide {slide} has missing label information")
            else:
                print(f"Slide {slide} not found")

    def __getitem__(self, idx):
        data0 = torch.load(self.data[idx].data_path)
        label0 = self.data[idx].label
        slide_id0 = self.data[idx].slide_id

        # If in training mode, apply augmentation with a certain probability
        if self.mode == 'train':
            if self.augmentation and np.random.rand() < self.aug_prob:
                # Randomly sample another slide for mixing
                sample = self.data[np.random.randint(len(self.data))]
                data1 = torch.load(sample.data_path)
                data1 = data1[np.random.choice(data1.shape[0], data0.shape[0], replace=True)]  # Sample from data1 to match the shape of data0
                label1 = sample.label
                w0 = np.random.rand()  # Weight for mixing two samples
                
                # Create augmented data by linear interpolation of two samples
                aug_data = w0 * data0 + (1 - w0) * data1
                aug_label = label_vec_generator(label0=label0, label1=label1, w0=w0, soft_labels=self.soft_labels, cls_count=self.cls_count)

                # Augment age and location information
                aug_age = age_augmentation(self.data[idx].age, sample.age, w0, self.drop_prob, self.dim_age_embed)
                aug_loc = loc_augmentation(self.data[idx].location, sample.location, w0, self.drop_prob, self.loc_dict)
                return aug_data, aug_age, aug_loc, aug_label, slide_id0
            else:
                # Get age and location representations if no augmentation is applied
                age0 = get_positional_encoding(self.data[idx].age, self.dim_age_embed)
                loc0 = get_loc_representation(self.data[idx].location, self.loc_dict)
                label0 = label_vec_generator(label0, soft_labels=self.soft_labels, cls_count=self.cls_count)
                return data0, age0, loc0, label0, slide_id0
        else:
            if self.soft_labels:
                label0 = torch.argmax(label0)
            age0 = get_positional_encoding(self.data[idx].age, self.dim_age_embed)
            loc0 = get_loc_representation(self.data[idx].location, self.loc_dict)
            return data0, age0, loc0, label0, slide_id0
        
    def __len__(self):
        return len(self.data)
    
    
class DataModule(pl.LightningDataModule):
    """
    DataModule for managing data loading and preparation for PyTorch Lightning training.

    Args:
        data_dir (str): Directory where the data files are stored.
        data_split (str): Path to YAML file containing train/test split information.
        batch_size (int): Batch size for training and validation. Defaults to 1.
        kwargs: Additional keyword arguments for data configuration.
    """       
    def __init__(self, data_dir, data_split, batch_size=1, **kwargs): 
        super().__init__()
        self.data_dir = data_dir
        self.data_split = read_yaml(data_split)  # Read train/test split information from YAML file
        self.labels = pd.read_csv(kwargs['label_file'])  # Load label information from CSV file

        # Load mapping from string labels to numerical labels
        self.str2label_mapping = read_yaml(kwargs['label_mapping'])['mapping']
        self.soft_labels = kwargs['soft_labels']  
        self.aug = kwargs['aug'] 

        # Parameters for augmentation
        self.aug_params = {'age_loc_drop_prob': kwargs['age_loc_drop_prob'], 'aug_prob': kwargs['aug_prob'], 
                           'cls_count': kwargs['cls_count']}
        
        self.loc_dict = kwargs['loc_dict']
        self.dim_age_embed = kwargs['dim_age_embed']  # Dimension for age embedding

        self.batch_size = int(batch_size)
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Set up datasets for different stages (train, validation, test)
        if stage == 'fit' or stage is None:
            self.train_dataset = self._initialize_dataset('train')
            self.val_dataset = self._initialize_dataset('test')

        if stage == 'test':
            self.test_dataset = self._initialize_dataset('test')
    
    def _initialize_dataset(self, split_key):
        # Helper function to initialize dataset for given split
        aug = self.aug if split_key == 'train' else False  # Apply augmentation only during training
        return LocalDataset(data_dir=self.data_dir, split=self.data_split[split_key], labels=self.labels, 
                            mapping=self.str2label_mapping, soft_labels=self.soft_labels, augmentation=aug,
                            loc_dict=self.loc_dict, dim_age_embed=self.dim_age_embed, mode=split_key, **self.aug_params)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)