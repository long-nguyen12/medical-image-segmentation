import os
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

datasets = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir-SEG"]

def move_images(image_list, source_path, destination_path):
    for image in image_list:
        source = os.path.join(source_path, image)
        mask_source = source.replace("images", "masks")
        destination = os.path.join(destination_path, image)
        mask_destination = destination.replace("images", "masks")
        shutil.move(source, destination)
        shutil.move(mask_source, mask_destination)

for dataset in datasets:
    origin_image_path = f'data/dataset/{dataset}'
    train_dir = f'{origin_image_path}/train/images/'
    validation_dir = f'{origin_image_path}/validation/images/'
    test_dir = f'{origin_image_path}/test/images/'

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    os.makedirs(train_dir.replace("images", "masks"), exist_ok=True)
    os.makedirs(validation_dir.replace("images", "masks"), exist_ok=True)
    os.makedirs(test_dir.replace("images", "masks"), exist_ok=True)

    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    all_images = os.listdir(f'{origin_image_path}/images/')
    train_images, temp_images = train_test_split(all_images, test_size=1 - train_ratio, random_state=42)
    validation_images, test_images = train_test_split(temp_images, test_size=test_ratio/(validation_ratio+test_ratio), random_state=42)

    move_images(train_images, f'{origin_image_path}/images/', train_dir)
    move_images(validation_images, f'{origin_image_path}/images/', validation_dir)
    move_images(test_images, f'{origin_image_path}/images/', test_dir)

