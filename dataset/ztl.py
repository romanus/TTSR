import os
from imageio import imread, imwrite
from PIL import Image, ImageFilter
import numpy as np
import copy
import random
import multiprocessing

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

src_rows, src_cols = 1916, 3364
dst_resolution = 256

g_mutex = multiprocessing.Lock()

def get_item(input_image_path, reference_image_path, mask_image_path, input_rect, reference_rect):

    # improve input_image to the quality of ref_image
    input_image = imread(input_image_path)
    ref_image = imread(reference_image_path)

    row1, col1, row2, col2 = input_rect
    HR = ref_image[row1:row2,col1:col2,:]
    LR_sr = np.copy(input_image[row1:row2,col1:col2,:])
    LR = np.array(Image.fromarray(LR_sr).resize((dst_resolution // 4, dst_resolution // 4), Image.BICUBIC))

    row1, col1, row2, col2 = reference_rect
    Ref = ref_image[row1:row2,col1:col2,:]
    Ref_sr = input_image[row1:row2,col1:col2,:]

    LR_sr[np.where(LR_sr == 0)] = 1 # reduce the active domain to [1, 255]
    LR[np.where(LR == 0)] = 1

    if mask_image_path is not None:
        mask = imread(mask_image_path)
        mask_sr = np.array(Image.fromarray(mask).resize((dst_resolution, dst_resolution), Image.BICUBIC))
        mask_lr = np.array(Image.fromarray(mask).resize((dst_resolution // 4, dst_resolution // 4), Image.BICUBIC))
        LR_sr[np.where(mask_sr == 0)] = 0
        LR[np.where(mask_lr == 0)] = 0

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

def get_aligned_pairs(dataset_path):

    aligned_pairs = []
    for dir in os.listdir(dataset_path):
        dataset_item_folder = os.path.join(dataset_path, dir, 'aligned')
        for i in range(1, 6):
            ref_image_path = os.path.join(dataset_item_folder, "0000{}.JPG".format(i))
            input_image_path = os.path.join(dataset_item_folder, "0000{}.JPG".format(i+1))

            if not (os.path.exists(ref_image_path) and os.path.exists(input_image_path)):
                break

            aligned_pairs.append((input_image_path, ref_image_path))

        for i in range(1, 5):
            ref_image_path = os.path.join(dataset_item_folder, "0000{}.JPG".format(i))
            input_image_path = os.path.join(dataset_item_folder, "0000{}.JPG".format(i+2))

            if not (os.path.exists(ref_image_path) and os.path.exists(input_image_path)):
                break

            aligned_pairs.append((input_image_path, ref_image_path))

    return aligned_pairs

def generate_aligned_rects(random_generator):
    ref_row_start = random_generator.randrange(src_rows - dst_resolution)
    ref_col_start = random_generator.randrange(src_cols - dst_resolution)

    row1, col1, row2, col2 = ref_row_start, ref_col_start, ref_row_start+dst_resolution, ref_col_start+dst_resolution
    input_rect = (row1, col1, row2, col2)

    input_row_start = random_generator.randrange(max(0, ref_row_start - (dst_resolution * 3) // 4), min(ref_row_start + (dst_resolution * 3) // 4, src_rows - dst_resolution))
    input_col_start = random_generator.randrange(max(0, ref_col_start - (dst_resolution * 3) // 4), min(ref_col_start + (dst_resolution * 3) // 4, src_cols - dst_resolution))

    row1, col1, row2, col2 = input_row_start, input_col_start, input_row_start+dst_resolution, input_col_start+dst_resolution

    reference_rect = row1, col1, row2, col2

    return (input_rect, reference_rect)

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

        self.mask_rate = args.mask_rate

        self.random_generator = random.Random(42)

        self.input_list = get_aligned_pairs(os.path.join(args.dataset_dir, 'train'))
        self.input_list = self.input_list[0:int(len(self.input_list) * args.dataset_rate)]

        masks_folder = os.path.join(args.dataset_dir, 'qd_imd', 'train')
        self.mask_paths_list = [os.path.join(masks_folder, filename) for filename in os.listdir(masks_folder)]
        self.random_generator.shuffle(self.mask_paths_list)
        self.mask_idx = 0

        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):

        input_image_path, reference_image_path = self.input_list[idx]

        with g_mutex:
            input_rect, reference_rect = generate_aligned_rects(self.random_generator)

            mask_image_path = None
            if self.random_generator.random() < self.mask_rate: # apply mask
                mask_image_path = self.mask_paths_list[self.mask_idx]
                self.mask_idx = (self.mask_idx + 1) % len(self.mask_paths_list)

        sample = get_item(input_image_path, reference_image_path, mask_image_path, input_rect, reference_rect)

        if self.transform:
            sample = self.transform(sample)

        return sample

class TestSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([ToTensor()])):

        self.random_generator = random.Random(42)

        self.input_list = get_aligned_pairs(os.path.join(args.dataset_dir, 'test'))
        self.random_generator.shuffle(self.input_list)
        self.input_list = self.input_list[0:int(len(self.input_list) * args.dataset_rate)]

        masks_folder = os.path.join(args.dataset_dir, 'qd_imd', 'test')
        masks_list = [os.path.join(masks_folder, filename) for filename in os.listdir(masks_folder)]
        self.random_generator.shuffle(masks_list)

        self.rects_list = []
        self.masks_list = []
        for mask_idx in range(len(self.input_list)):

            # append to self.rects_list
            self.rects_list.append(generate_aligned_rects(self.random_generator))

            # append to self.masks_list
            if self.random_generator.random() < args.mask_rate: # apply mask
                self.masks_list.append(masks_list[mask_idx])
            else:
                self.masks_list.append(None)

        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input_image_path, reference_image_path = self.input_list[idx]
        mask_image_path = self.masks_list[idx]
        input_rect, reference_rect = self.rects_list[idx]
        sample = get_item(input_image_path, reference_image_path, mask_image_path, input_rect, reference_rect)

        if self.transform:
            sample = self.transform(sample)

        return sample