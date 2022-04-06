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

def get_item(idx, input_image_path, ref_image_path, gt_image_path):
    ### HR
    HR = imread(gt_image_path)

    ### LR and LR_sr
    LR_sub = imread(input_image_path)
    h,w = LR_sub.shape[:2]
    LR = np.array(Image.fromarray(LR_sub).resize((w//4, h//4), Image.BICUBIC))
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

def train_test_split(path, train_ratio=0.95, dataset_part=0.6):
    images_list = []

    for root, _, files in os.walk(path):
        for file in files:
            input_file_path = os.path.join(root, file)
            images_list.append(input_file_path)

    if dataset_part != 1.0:
        images_list = images_list[0:int(len(images_list)*dataset_part)]

    all_images_len = len(images_list)
    train_images_len = int(train_ratio * all_images_len)
    test_images_len = all_images_len - train_images_len

    return torch.utils.data.random_split(images_list, [train_images_len, test_images_len], generator=torch.Generator().manual_seed(42))

class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):

        dataset_dir_src = os.path.join(args.dataset_dir, 'src')
        dataset_dir_ref = os.path.join(args.dataset_dir, 'ref')
        dataset_dir_gt = os.path.join(args.dataset_dir, 'ground-truth')

        train_set, _ = train_test_split(dataset_dir_src)
        train_set = list(train_set)

        self.input_list = train_set
        self.ref_list = copy.deepcopy(train_set)
        self.gt_list = copy.deepcopy(train_set)

        for i in range(len(train_set)):
            self.ref_list[i] = self.ref_list[i].replace(dataset_dir_src, dataset_dir_ref)
            self.gt_list[i] = self.gt_list[i].replace(dataset_dir_src, dataset_dir_gt)

        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        sample = get_item(idx, self.input_list[idx], self.ref_list[idx], self.gt_list[idx])

        if self.transform:
            sample = self.transform(sample)
        return sample


class TestSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):

        dataset_dir_src = os.path.join(args.dataset_dir, 'src')
        dataset_dir_ref = os.path.join(args.dataset_dir, 'ref')
        dataset_dir_gt = os.path.join(args.dataset_dir, 'ground-truth')

        _, test_set = train_test_split(dataset_dir_src)
        test_set = list(test_set)

        self.input_list = test_set
        self.ref_list = copy.deepcopy(test_set)
        self.gt_list = copy.deepcopy(test_set)

        for i in range(len(test_set)):
            self.ref_list[i] = self.ref_list[i].replace(dataset_dir_src, dataset_dir_ref)
            self.gt_list[i] = self.gt_list[i].replace(dataset_dir_src, dataset_dir_gt)

        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        sample = get_item(idx, self.input_list[idx], self.ref_list[idx], self.gt_list[idx])

        if self.transform:
            sample = self.transform(sample)
        return sample