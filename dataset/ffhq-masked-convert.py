import os
from imageio import imread, imwrite
from PIL import Image, ImageFilter
import numpy as np
import random

resolution = 160

if __name__ == '__main__':
    input_folder = os.path.join(".", "ffhq{}".format(resolution))
    input_mask_folder = os.path.join(".", "qd_imd", "train")
    output_folder = os.path.join(".", "ffhq{}-masked".format(resolution))
    output_folder_src = os.path.join(output_folder, 'src')
    output_folder_ref = os.path.join(output_folder, 'ref')
    output_folder_gt = os.path.join(output_folder, 'ground-truth')

    assert(os.path.exists(input_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(output_folder_src)
        os.makedirs(output_folder_ref)
        os.makedirs(output_folder_gt)

    masks_files = os.listdir(input_mask_folder)
    random.seed(42)
    def get_mask_file():
        return masks_files[random.randrange(len(masks_files))]

    for root, directories, files in os.walk(input_folder):
        for file in files:
            input_file_path = os.path.join(root, file)
            output_file_path_src = os.path.join(output_folder_src, file)
            output_file_path_ref = os.path.join(output_folder_ref, file)
            output_file_path_gt = os.path.join(output_folder_gt, file)
            input_mask_path = os.path.join(input_mask_folder, get_mask_file())

            if not os.path.exists(output_file_path_gt):
                input_image = imread(input_file_path)
                input_mask = imread(input_mask_path)

                input_mask = np.array(Image.fromarray(input_mask).resize((resolution, resolution), Image.BICUBIC))
                input_mask = np.reshape(input_mask, (resolution, resolution, 1))
                input_mask = np.concatenate((input_mask, input_mask, input_mask), axis=2)

                output_image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
                background_mask = np.where(input_mask != 0)
                foreground_mask = np.where(input_mask == 0)

                output_image[background_mask] = input_image[background_mask] # ref with black areas
                imwrite(output_file_path_ref, output_image)

                input_image_blurred = np.array(Image.fromarray(input_image).filter(ImageFilter.BoxBlur(5)))
                output_image[foreground_mask] = input_image_blurred[foreground_mask] # src with blurred areas
                imwrite(output_file_path_src, output_image)

                imwrite(output_file_path_gt, input_image)