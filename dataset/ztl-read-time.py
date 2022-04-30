import os
from imageio import imread
from tqdm import tqdm

if __name__ == '__main__':
    input_folder = os.path.join(".", "ztl", "train")

    min_rows, min_cols = 10000, 10000

    for dir in tqdm(os.listdir(input_folder), ascii=True):

        input_item_folder = os.path.join(input_folder, dir, 'aligned')
        for i in range(1, 6):

            ref_image_path = os.path.join(input_item_folder, "0000{}.JPG".format(i))
            input_image_path = os.path.join(input_item_folder, "0000{}.JPG".format(i+1))

            if not (os.path.exists(ref_image_path) and os.path.exists(input_image_path)):
                break

            ref_image = imread(ref_image_path)
            input_image = imread(input_image_path)

            rows, cols, _ = ref_image.shape
            min_rows, min_cols = min(min_rows, rows), min(min_cols, cols)
            rows, cols, _ = input_image.shape
            min_rows, min_cols = min(min_rows, rows), min(min_cols, cols)

    print((min_rows, min_cols))