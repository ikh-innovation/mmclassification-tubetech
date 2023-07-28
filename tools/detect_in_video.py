from pathlib import Path
import os
import cv2
import platform
import numpy as np
from mmpretrain import ImageClassificationInferencer
import time


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


class LoadVideo:
    """
    A class for loading video frames.

    Args:
        path (str): Path to the video file.
        vid_stride (int): Frame-rate stride of the video (default: 1).
        starting_frame (int): The frame number to start from (default: None).
    """

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
        for _ in range(self.vid_stride):
            self.cap.grab()
        ret_val, im0 = self.cap.retrieve()

        self.frame += 1
        s = f'video ({self.frame}/{self.frames}) {self.path}: '
        return im0, s

    def _new_video(self):
        self.frame = 0
        self.cap = cv2.VideoCapture(self.path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))


    def _go_to_frame(self, frame_no):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        print('video Position set: ', int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))


class TileImage:
    """
    A class for tiling and reconstructing images, in order to apply a model or visual functions in each tile.

    Args:
        img (numpy.ndarray): Input image.
        n_tiles (int): Number of tiles to be split (default: 4).
    """

    def __init__(self, img, n_tiles=4):
        self.img = img
        self.n_tiles = int(n_tiles // 2) if n_tiles > 1 else 1
        self.tile_shape = np.array(img.shape) // self.n_tiles
        self.tiles = None
        self.reconstructed_image = None

    def _make_tiles(self):
        # split into sqrt(tiles)
        self.tiles = self.img.reshape(self.n_tiles, self.tile_shape[0], self.n_tiles, self.tile_shape[1], 3)
        self.tiles = np.transpose(self.tiles, (0, 2, 1, 3, 4))
        self.tiles = self.tiles.reshape(-1, self.tile_shape[0], self.tile_shape[1], 3)

    def _reconstruct_tiled_image(self):
        ''' reconstruct image from tiles'''

        recon_img = self.tiles.reshape(self.n_tiles, self.n_tiles, self.tile_shape[0], self.tile_shape[1], 3)
        recon_img = np.transpose(recon_img, (0, 2, 1, 3, 4))
        recon_img = recon_img.reshape( *self.img.shape)
        self.reconstructed_image = recon_img

    def _add_classification_visuals(self, img, pred_class, conf: float, correct=True, threshold=None, frame_thickness=10):

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
            cv2.putText(img=img, text="threshold: " + str(round(threshold, 2)), org=(50, 180),
                        fontFace=3, fontScale=1.5, color=(255, 0, 0), thickness=2)

    def infer_tiles(self, model, conf_thresh = 0.8):
        """
        Infers predictions on each tile using the provided model.

        Args:
            model: The image classification model.(mm inferencer type).
            conf_thresh (float): Confidence threshold (default: 0.8).

        Returns:
            numpy.ndarray: Reconstructed image with classification visuals on each tile.
        """
        self._make_tiles()
        tile_list = [t for t in self.tiles]
        results = model(tile_list)

        for t, r in zip(tile_list, results):

            # classified as correct/clean only if it passed a threshold of confidence
            # correct = not bool(r['pred_label']) and (r['pred_score'] > conf_thresh)

            # pure model prediction
            correct = not bool(r['pred_label'])

            self._add_classification_visuals(t, pred_class=r['pred_class'], conf=r['pred_score'],
                                       correct=correct) #  threshold=conf_thresh

        self._reconstruct_tiled_image()
        return self.reconstructed_image


class VideoTileInferencer:
    """
    A class for inferring predictions on tiles of a video.

    Args:
        video_path (str): Path to the input video.
        config (str): Path to the model configuration file.
        checkpoint (str): Path to the model checkpoint file.
        video_export_path (str): Path to export the processed video. If a path is given, then the video will be exported there. (default: None).
        n_tiles (int): Number of tiles to use (default: 4).
        vid_stride (int): Frame-rate stride of the video (default: 1).
        starting_frame (int): The frame number to start from (default: None).
        show (bool): Whether to display the processed video (default: True).
    """
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

            ####import time

            start = time.time()

            tile_image = TileImage(im, self.n_tiles)
            recon_img = tile_image.infer_tiles(self.inferencer)

            el_time = time.time() - start
            print(" inference time (", self.n_tiles," tiles): ", round(el_time, 2), " fps: ", round(1/el_time, 2) )
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


if __name__ == '__main__':
    video_path = "../../boiler_unit_9.mp4"
    video_export_path = "../../boiler_unit_9_prediction_tiled_v1.avi"
    # video_export_path = None
    # image = 'https://github.com/open-mmlab/mmpretrain/raw/main/demo/demo.JPEG'
    model_folder = "../work_dirs/efficientnet-b5_2xb4_in1k-456px_boiler_defects_tiled_v1/"
    # config = model_folder + "efficientnet-b5_2xb4_in1k-456px_boiler_defects.py"
    config = "../configs/efficientnet/efficientnet-b5_2xb4_in1k-456px_boiler_defects_tiled_v1.py"
    checkpoint = model_folder + "best_multi-label_precision_top1_epoch_42.pth"

    videoTileInferencer = VideoTileInferencer(video_path, config, checkpoint, video_export_path=video_export_path, n_tiles=4, vid_stride=1, starting_frame=1500, show=True)
    videoTileInferencer.start()
