from pathlib import Path
import os
import cv2
import platform
import numpy as np
from mmpretrain import ImageClassificationInferencer


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


class LoadVideo:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, transforms=None, vid_stride=1):
        assert path.split('.')[-1].lower() in VID_FORMATS, 'not supported video format or wrong path'
        self.path = path
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        self._new_video()

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        # Read video
        for _ in range(self.vid_stride):
            self.cap.grab()
        ret_val, im0 = self.cap.retrieve()

        self.frame += 1
        # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
        s = f'video ({self.frame}/{self.frames}) {self.path}: '
        # im = im0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # im = im0
        return im0, s

    def _new_video(self):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(self.path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im


class TileImage:
    """
    Args:
        img: numpy array
        n_tiles: has to be in power of 2.
    """

    def __init__(self, img, n_tiles=4):
        self.img = img
        self.n_tiles = np.sqrt(n_tiles).astype(int)
        self.tile_shape = np.array(im.shape) // self.n_tiles
        self.tiles = None
        self.reconstructed_image = None


    def make_tiles(self):
        # split into sqrt(tiles)
        self.tiles = self.img.reshape(self.n_tiles, self.tile_shape[0], self.n_tiles, self.tile_shape[1], 3)
        self.tiles = np.transpose(self.tiles, (0, 2, 1, 3, 4))
        self.tiles = self.tiles.reshape(-1, self.tile_shape[0], self.tile_shape[1], 3)

    def reconstruct_tiled_image(self):
        ''' reconstruct image from tiles'''

        recon_img = self.tiles.reshape(self.n_tiles, self.n_tiles, self.tile_shape[0], self.tile_shape[1], 3)
        recon_img = np.transpose(recon_img, (0, 2, 1, 3, 4))
        recon_img = recon_img.reshape( *self.img.shape)
        self.reconstructed_image = recon_img

    def infer_tiles(self, model):
        def add_box(img, color=[0, 255, 0]):
            img[:, :10, :] = color
            img[:, -10:, :] = color
            img[:10, :, :] = color
            img[-10:, :, :] = color
            return img

        self.make_tiles()

        tile_list = [t for t in self.tiles]

        results = model(tile_list)

        for t, r in zip(tile_list, results):
            if not bool(r['pred_label']):
                add_box(t, [0, 255, 0])
            else:
                add_box(t, [0, 0, 255])


        self.reconstruct_tiled_image()
        return self.reconstructed_image


video_path = "../../boiler_unit_9.mp4"
# image = 'https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG'
model_folder = "../work_dirs/efficientnet-b5_2xb4_in1k-456px_boiler_defects/"
# config = model_folder + "efficientnet-b5_2xb4_in1k-456px_boiler_defects.py"
config = "../configs/efficientnet/efficientnet-b5_2xb4_in1k-456px_boiler_defects.py"
checkpoint = model_folder + "best_accuracy_top1_epoch_53.pth"
inferencer = ImageClassificationInferencer(model=config, pretrained=checkpoint, device='cuda')


# Stream results
vid_iter = LoadVideo(video_path)

first_frame, _ = next(vid_iter)
if platform.system() == 'Linux':
    cv2.namedWindow(str(video_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    cv2.resizeWindow(str(video_path), first_frame.shape[1], first_frame.shape[0])


for  i, (im, s) in enumerate(vid_iter):
    if i <1200: continue
    # im0 = annotator.result()
    tile_image = TileImage(im)
    recon_img = tile_image.infer_tiles(inferencer)


    cv2.imshow(str(video_path), recon_img)
    print(s)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

