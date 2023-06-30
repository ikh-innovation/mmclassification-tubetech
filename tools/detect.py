from pathlib import Path
import os
import cv2
import platform
import numpy as np


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

def add_box(img, color=[0,255,0]):
    img[:, :10, :] = color
    img[:, -10:, :] = color
    img[:10, :, :] = color
    img[ -10  :, :, :] = color

    return img

def split(img, n_tiles=4):
    tile_shape = np.array(im.shape)//n_tiles

    tiles = img.reshape(n_tiles,tile_shape[0],n_tiles, tile_shape[1], 3)
    tiles = np.transpose(tiles, (0, 2, 1, 3, 4))
    tiles = tiles.reshape(-1, tile_shape[0], tile_shape[1], 3)

    for t in tiles:
        add_box(t)
        cv2.namedWindow("test", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        cv2.resizeWindow("test", img.shape[0], img.shape[1])
        cv2.imshow("test", t)
        # cv2.imshow("awd", img)
        cv2.waitKey(7200)
    imgr = tiles.reshape(img.shape)

    cv2.namedWindow("test", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    cv2.resizeWindow("test", img.shape[0], img.shape[1])
    cv2.imshow("test", imgr)
    # cv2.imshow("awd", img)
    cv2.waitKey(7200)

    # for x_tile in np.split(n_tiles, img):
    #     for y_tile in  np.split(n_tiles, x_tile, axis=1):
    #         print(0)
    return 0




# Stream results

video_path = "../../boiler_unit_9.mp4"
vid_iter = LoadVideo(video_path)

if platform.system() == 'Linux':
    cv2.namedWindow(str(video_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    # cv2.resizeWindow(str(video_path), im.shape[1], im.shape[0])

for (im, s) in vid_iter:
    # im0 = annotator.result()
    split(im)
    cv2.imshow(str(video_path), im)
    print(s)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

