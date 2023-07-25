from pathlib import Path
import os
import shutil
import random


def split_dataset(source_dir, target_dir, train_percentage=0.7, valid_percentage=0.15, classes=None):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get the list of classes in the source directory
    if not classes:
        classes = os.listdir(source_dir)

    # Iterate over each class
    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        if os.path.isdir(class_dir):
            # Create the train directory for the current class
            train_class_dir = os.path.join(target_dir, "train", class_name)
            os.makedirs(train_class_dir)

            # Create the validation directory for the current class
            valid_class_dir = os.path.join(target_dir, "valid", class_name)
            os.makedirs(valid_class_dir)\

            # Create the test directory for the current class
            test_class_dir = os.path.join(target_dir, "test", class_name)
            os.makedirs(test_class_dir)

            # Get the list of images in the current class directory
            images = os.listdir(class_dir)
            random.shuffle(images)  # Shuffle the images

            # Calculate the number of training images based on the train_percentage
            num_train_images = int(len(images) * train_percentage)
            # Calculate the number of validation images based on the valid_percentage
            num_valid_images = int(len(images) * valid_percentage)

            # Iterate over each image
            for i, image_name in enumerate(images):
                src_path = os.path.join(class_dir, image_name)
                if i < num_train_images:
                    # Copy the image to the train directory
                    dst_path = os.path.join(train_class_dir, image_name)
                elif i < num_train_images + num_valid_images:
                    # Copy the image to the validation directory
                    dst_path = os.path.join(valid_class_dir, image_name)
                else:
                    # Copy the image to the test directory
                    dst_path = os.path.join(test_class_dir, image_name)

                shutil.copy(src_path, dst_path)


if __name__ == '__main__':

    root_path = Path("/home/innovation/Projects/tubetech")
    data_path = root_path/"tubetech_video_dataset_tiled"
    new_dataset_path = root_path/"boiler_defects_dataset_tiled_v1"

    subfolders = ["defect", "clean"]

    split_dataset(data_path, new_dataset_path, train_percentage=0.7, valid_percentage=0.2, classes=subfolders)



