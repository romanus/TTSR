import os
from imageio import imread, imwrite
from PIL import Image, ImageFilter
import numpy as np
import random
from tqdm import tqdm

# dst HR resolution
dst_resolution = 256

# 60% of files are masked
mask_rate = 0.6

if __name__ == '__main__':
    input_folder = os.path.join(".", "ztl", "train")
    input_mask_folder = os.path.join(".", "qd_imd", "train")
    output_folder = os.path.join(".", "ztl{}-masked".format(dst_resolution))

    assert(os.path.exists(input_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    masks_files = os.listdir(input_mask_folder)
    random.seed(42)
    random.shuffle(masks_files)
    mask_idx = 0

    for dir in tqdm(os.listdir(input_folder), ascii=True):

        input_item_folder = os.path.join(input_folder, dir, 'aligned')
        for i in range(1, 6):

            ref_image_path = os.path.join(input_item_folder, "0000{}.JPG".format(i))
            input_image_path = os.path.join(input_item_folder, "0000{}.JPG".format(i+1))

            if not (os.path.exists(ref_image_path) and os.path.exists(input_image_path)):
                break

            ref_image = imread(ref_image_path)
            input_image = imread(input_image_path)

            for j in range(3):
                output_item_folder = os.path.join(output_folder, "set{}_img{}_part{}".format(dir, i, j))
                if not os.path.exists(output_item_folder):
                    os.makedirs(output_item_folder)
                dst_ref_image_path = os.path.join(output_item_folder, "ref.png")
                dst_ref_sr_image_path = os.path.join(output_item_folder, "ref_sr.png")
                dst_input_image_path = os.path.join(output_item_folder, "src.png")
                dst_input_lr_image_path = os.path.join(output_item_folder, "src_lr.png")
                dst_input_sr_image_path = os.path.join(output_item_folder, "src_sr.png")

                assert(input_image.shape == ref_image.shape)
                (rows, cols, _) = ref_image.shape
                ref_row_start = random.randrange(rows - dst_resolution)
                ref_col_start = random.randrange(cols - dst_resolution)

                row1, col1, row2, col2 = ref_row_start, ref_col_start, ref_row_start+dst_resolution, ref_col_start+dst_resolution

                dst_ref_image = ref_image[row1:row2,col1:col2,:]
                dst_ref_sr_image = input_image[row1:row2,col1:col2,:]

                input_row_start = random.randrange(max(0, ref_row_start - (dst_resolution * 3) // 4), min(ref_row_start + (dst_resolution * 3) // 4, rows - dst_resolution))
                input_col_start = random.randrange(max(0, ref_col_start - (dst_resolution * 3) // 4), min(ref_col_start + (dst_resolution * 3) // 4, cols - dst_resolution))

                row1, col1, row2, col2 = input_row_start, input_col_start, input_row_start+dst_resolution, input_col_start+dst_resolution

                dst_input_image = ref_image[row1:row2,col1:col2,:] # ground truth
                dst_input_sr_image = np.copy(input_image[row1:row2,col1:col2,:])
                dst_input_lr_image = np.array(Image.fromarray(dst_input_sr_image).resize((dst_resolution // 4, dst_resolution // 4), Image.BICUBIC))

                dst_input_sr_image[np.where(dst_input_sr_image == 0)] = 1 # reduce the active domain to [1, 255]
                dst_input_lr_image[np.where(dst_input_lr_image == 0)] = 1

                if random.random() < mask_rate: # apply mask
                    mask = imread(os.path.join(input_mask_folder, masks_files[mask_idx]))
                    mask_idx += 1
                    mask_sr = np.array(Image.fromarray(mask).resize((dst_resolution, dst_resolution), Image.BICUBIC))
                    mask_lr = np.array(Image.fromarray(mask).resize((dst_resolution // 4, dst_resolution // 4), Image.BICUBIC))
                    dst_input_sr_image[np.where(mask_sr == 0)] = 0
                    dst_input_lr_image[np.where(mask_lr == 0)] = 0

                imwrite(dst_ref_image_path, dst_ref_image)
                imwrite(dst_ref_sr_image_path, dst_ref_sr_image)
                imwrite(dst_input_image_path, dst_input_image)
                imwrite(dst_input_lr_image_path, dst_input_lr_image)
                imwrite(dst_input_sr_image_path, dst_input_sr_image)

    # for root, directories, files in os.walk(input_folder):
    #     for file in files:
    #         input_file_path = os.path.join(root, file)
    #         output_file_path_src = os.path.join(output_folder_src, file)
    #         output_file_path_gt = os.path.join(output_folder_gt, file)
    #         input_mask_path = os.path.join(input_mask_folder, get_mask_file())

    #         if not os.path.exists(output_file_path_gt):
    #             input_image = imread(input_file_path)
    #             input_mask = imread(input_mask_path)

    #             input_image[np.where(input_image == 0)] = 1 # reduce the active domain to [1, 255]

    #             input_mask = np.array(Image.fromarray(input_mask).resize((resolution, resolution), Image.BICUBIC))
    #             input_mask = np.reshape(input_mask, (resolution, resolution, 1))
    #             input_mask = np.concatenate((input_mask, input_mask, input_mask), axis=2)

    #             dropout = random.randrange(10) > 1 # 80% of images get degraded
    #             if not dropout:
    #                 output_image = input_image
    #             else:
    #                 output_image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    #                 background_mask = np.where(input_mask != 0)
    #                 output_image[background_mask] = input_image[background_mask] # ref with black areas
    #             imwrite(output_file_path_src, output_image)

    #             # foreground_mask = np.where(input_mask == 0)
    #             # input_image_blurred = np.array(Image.fromarray(input_image).filter(ImageFilter.BoxBlur(5)))
    #             # output_image[foreground_mask] = input_image_blurred[foreground_mask] # src with blurred areas
    #             # imwrite(output_file_path_src, output_image)

    #             imwrite(output_file_path_gt, input_image)