from pathlib import Path
import os
import cv2
import platform
import numpy as np
from mmpretrain import ImageClassificationInferencer


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


class LoadVideo:
    # image/video dataloader similar(stolen) with Yolov5.

    def __init__(self, path, vid_stride=1, starting_frame=None):
        assert path.split('.')[-1].lower() in VID_FORMATS, 'not supported video format or wrong path'
        self.path = path
        self.vid_stride = vid_stride  # video frame-rate stride
        self._new_video()
        if starting_frame:
            self._go_to_frame(starting_frame)

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
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
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

    def _go_to_frame(self, frame_no):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        print('video Position set: ', int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))


class TileImage:
    """
    Args:
        img: numpy array
    """

    def __init__(self, img, n_tiles=4):
        self.img = img
        self.n_tiles = int(n_tiles//2) if n_tiles>1 else 1
        self.tile_shape = np.array(img.shape) // self.n_tiles
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

    def infer_tiles(self, model, conf_thresh = 0.8):
        def add_classification_visuals(img, pred_class, conf: float, correct=True,  threshold=None, frame_thickness=10):

            color = (0, 255, 0) if correct else (0, 0, 255)

            # add color frame
            img[:, :frame_thickness, :] = color
            img[:, -frame_thickness:, :] = color
            img[:frame_thickness, :, :] = color
            img[-frame_thickness:, :, :] = color

            # add text
            cv2.putText(img=img, text="confidence: " + str(round(conf, 2)), org=(50, 80),
                        fontFace=3, fontScale=1.5, color=color, thickness=2)

            cv2.putText(img=img, text="class: " + pred_class, org=(50, 130),
                        fontFace=3, fontScale=1.5, color=color, thickness=2)

            if threshold:
                cv2.putText(img=img, text="threshold: " +  str(round(threshold, 2)), org=(50, 180),
                        fontFace=3, fontScale=1.5, color=(255,0,0), thickness=2)


        self.make_tiles()

        tile_list = [t for t in self.tiles]

        results = model(tile_list)

        for t, r in zip(tile_list, results):

            # classified as correct/clean only if it passed a threshold of confidence
            # correct = not bool(r['pred_label']) and (r['pred_score'] > conf_thresh)

            # pure model prediction
            correct = not bool(r['pred_label'])

            add_classification_visuals(t, pred_class=r['pred_class'], conf=r['pred_score'],
                                       correct=correct) #  threshold=conf_thresh

        self.reconstruct_tiled_image()
        return self.reconstructed_image


class VideoTileInferencer:

    def __init__(self, video_path, config, checkpoint, video_export_path=None, n_tiles=4, vid_stride=1, starting_frame=None, show=True):
        self.video_path = video_path
        self.video_export_path = video_export_path
        self.config = config
        self.checkpoint = checkpoint
        self.n_tiles = n_tiles
        self.vid_stride = vid_stride
        self.show = show

        self.vid_iter = LoadVideo(self.video_path, self.vid_stride, starting_frame)
        first_frame, _ = next(self.vid_iter)

        self.inferencer = ImageClassificationInferencer(model=config, pretrained=checkpoint, device='cuda')

        if platform.system() == 'Linux' and show:
            cv2.namedWindow(str(self.video_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(self.video_path), first_frame.shape[1], first_frame.shape[0])

        if self.video_path:
            self.video_export = cv2.VideoWriter(video_export_path, cv2.VideoWriter_fourcc(*'MJPG'), self.vid_iter.fps//self.vid_stride,
                                                (first_frame.shape[1], first_frame.shape[0]))

    def start(self):
        for i, (im, s) in enumerate(self.vid_iter):

            ####
            tile_image = TileImage(im, self.n_tiles)
            recon_img = tile_image.infer_tiles(self.inferencer)

            print(s)

            if self.video_path:
                self.video_export.write(recon_img)
            if self.show:
                cv2.imshow(str(video_path), recon_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.vid_iter.cap.release()
                if video_path:
                    self.video_export.release()
                    print("The video was successfully saved")
                # Closes all the frames
                cv2.destroyAllWindows()
                break


video_path = "../../boiler_unit_9.mp4"
video_export_path = "../../boiler_unit_9_prediction.avi"
# image = 'https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG'
model_folder = "../work_dirs/efficientnet-b5_2xb4_in1k-456px_boiler_defects/"
# config = model_folder + "efficientnet-b5_2xb4_in1k-456px_boiler_defects.py"
config = "../configs/efficientnet/efficientnet-b5_2xb4_in1k-456px_boiler_defects.py"
checkpoint = model_folder + "best_accuracy_top1_epoch_53.pth"

videoTileInferencer = VideoTileInferencer(video_path, config, checkpoint, video_export_path=video_export_path, n_tiles=8, vid_stride=15, starting_frame=1500, show=True)
videoTileInferencer.start()
