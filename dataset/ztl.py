import os
from imageio import imread
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def get_item(input_image_path, ref_image_path):
    ### HR
    HR = imread(input_image_path)
    h,w = HR.shape[:2]
    #HR = HR[:h//4*4, :w//4*4, :]

    ### LR and LR_sr
    LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC))
    LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

    ### Ref and Ref_sr
    Ref_sub = imread(ref_image_path)
    h2, w2 = Ref_sub.shape[:2]
    Ref_sr_sub = np.array(Image.fromarray(Ref_sub).resize((w2//4, h2//4), Image.BICUBIC))
    Ref_sr_sub = np.array(Image.fromarray(Ref_sr_sub).resize((w2, h2), Image.BICUBIC))

    ### complete ref and ref_sr to the same size, to use batch_size > 1
    Ref = np.zeros((h, w, 3))
    Ref_sr = np.zeros((h, w, 3))
    Ref[:h2, :w2, :] = Ref_sub
    Ref_sr[:h2, :w2, :] = Ref_sr_sub

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


class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):
        data_path = os.path.join(args.dataset_dir, 'test')
        data_folders = os.listdir(data_path)

        self.input_list = []
        self.ref_list = []

        for folder in data_folders:
            for idx in [7, 5, 4, 3]:
                input_image_path = os.path.join(data_path, folder, "0000{}.JPG".format(idx))
                ref_image_path = os.path.join(data_path, folder, "0000{}.JPG".format(idx-2))
                if(os.path.exists(input_image_path)):
                    self.input_list.append(input_image_path)
                    self.ref_list.append(ref_image_path)

        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        sample = get_item(self.input_list[idx], self.ref_list[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class TestSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):
        data_path = os.path.join(args.dataset_dir, 'test')
        data_folders = os.listdir(data_path)

        self.input_list = []
        self.ref_list = []

        for folder in data_folders:
            for idx in [6]:
                input_image_path = os.path.join(data_path, folder, "0000{}.JPG".format(idx))
                ref_image_path = os.path.join(data_path, folder, "0000{}.JPG".format(idx-2))
                if(os.path.exists(input_image_path)):
                    self.input_list.append(input_image_path)
                    self.ref_list.append(ref_image_path)

        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        sample = get_item(self.input_list[idx], self.ref_list[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample