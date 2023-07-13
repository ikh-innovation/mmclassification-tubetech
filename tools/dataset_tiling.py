from pathlib import Path
import os
from detect_in_video import TileImage
import cv2


if __name__ == '__main__':

    dataset_path = Path("/home/innovation/Projects/tubetech/tubetech_video_dataset")

    subfolders = os.listdir(dataset_path)

    # create new file structure
    new_dataset_path = Path(str(dataset_path) + "_tiled")

    for s in subfolders:
        os.makedirs(new_dataset_path/s, exist_ok=True)

        image_names = os.listdir(dataset_path/s)

        for im in image_names:
            img = cv2.imread(str(dataset_path/s/im))  # reads an image in the BGR format
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            tiled_im = TileImage(img, 4)
            tiled_im._make_tiles()

            im_name, stem = im.split('.')

            for i, tile in enumerate(tiled_im.tiles):
                new_im_name = im_name + "_T" + str(i+1) + "." + stem
                cv2.imwrite(str(new_dataset_path/s/new_im_name), tile)
            print("completed with ", im)




