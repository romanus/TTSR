import os
from imageio import imread, imwrite
from PIL import Image, ImageFilter
import numpy as np
import copy

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def get_item(input_folder_path):
    ### HR
    HR = imread(os.path.join(input_folder_path, 'src.png'))

    ### LR and LR_sr
    LR = imread(os.path.join(input_folder_path, 'src_lr.png'))
    LR_sr = imread(os.path.join(input_folder_path, 'src_sr.png'))

    ### Ref and Ref_sr
    Ref = imread(os.path.join(input_folder_path, 'ref.png'))
    Ref_sr = imread(os.path.join(input_folder_path, 'ref_sr.png'))

    ### change type
    LR = LR.astype(np.float32)
    LR_sr = LR_sr.astype(np.float32)
    HR = HR.astype(np.float32)
    Ref = Ref.astype(np.float32)
    Ref_sr = Ref_sr.astype(np.float32)

    ### rgb range to [-1, 1]
    LR = LR / 127.5 - 1.
    LR_sr = LR_sr / 127.5 - 1.
    HR = HR / 127.5 - 1.
    Ref = Ref / 127.5 - 1.
    Ref_sr = Ref_sr / 127.5 - 1.

    sample = {'LR': LR,
              'LR_sr': LR_sr,
              'HR': HR,
              'Ref': Ref,
              'Ref_sr': Ref_sr}

    return sample

class ToTensor(object):
    def __call__(self, sample):
        LR, LR_sr, HR, Ref, Ref_sr = sample['LR'], sample['LR_sr'], sample['HR'], sample['Ref'], sample['Ref_sr']
        LR = LR.transpose((2,0,1))
        LR_sr = LR_sr.transpose((2,0,1))
        HR = HR.transpose((2,0,1))
        Ref = Ref.transpose((2,0,1))
        Ref_sr = Ref_sr.transpose((2,0,1))
        return {'LR': torch.from_numpy(LR).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float()}

def train_test_split(path, train_ratio=0.9, dataset_part=0.1):
    dirs_list = []

    for folder in os.listdir(path):
        dirs_list.append(os.path.join(path, folder))

    if dataset_part != 1.0:
        dirs_list = dirs_list[0:int(len(dirs_list)*dataset_part)]

    all_dirs_len = len(dirs_list)
    train_dirs_len = int(train_ratio * all_dirs_len)
    test_dirs_len = all_dirs_len - train_dirs_len

    return torch.utils.data.random_split(dirs_list, [train_dirs_len, test_dirs_len], generator=torch.Generator().manual_seed(42))

class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):

        train_set, _ = train_test_split(args.dataset_dir)

        self.input_list = list(train_set)

        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        sample = get_item(self.input_list[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

class TestSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):

        _, test_set = train_test_split(args.dataset_dir)

        self.input_list = list(test_set)

        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        sample = get_item(self.input_list[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample