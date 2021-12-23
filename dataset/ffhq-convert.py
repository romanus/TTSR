import os
from imageio import imread, imwrite
from PIL import Image
import numpy as np

input_resolution = 1024
output_resolution = 160

if __name__ == '__main__':
    input_folder = os.path.join(".", "ffhq{}".format(input_resolution))
    output_folder = os.path.join(".", "ffhq{}".format(output_resolution))

    assert(os.path.exists(input_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    i = 0
    for root, directories, files in os.walk(input_folder):

        for directory in directories:
            dir_path = os.path.join(output_folder, directory)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        for file in files:
            input_file_path = os.path.join(root, file)
            output_file_path = input_file_path.replace(input_folder, output_folder)

            if not os.path.exists(output_file_path):
                input_image = imread(input_file_path)

                output_image = np.array(Image.fromarray(input_image).resize((output_resolution, output_resolution), Image.BICUBIC))

                imwrite(output_file_path, output_image)