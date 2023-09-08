from data.loaders import DataLoaderVideo
from utils.visualizers import Visualizer, grid_images
from utils.depth import render_depth, get_distance_at_point
from utils.logger import LOGGER
from models.dinov2_loader import *
import matplotlib.pyplot as plt

import cv2
import numpy as np
class Cursor:
    x = 0
    y = 0
    clicked = False
    @classmethod
    def set_cursor_position(cls, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cls.x = x
            cls.y = y
            cls.clicked = True

    @property
    def position(self):
        return (self.x, self.y)


if __name__=="__main__":
    LOGGER.info(f"Loading model...")
    loader = DinoV2Loader("small", "dpt", "nyu")
    depth_model = loader.load_model()
    LOGGER.info(f"Model loaded")

    imgsz = (720,1280)
    # Center Point
    center = [imgsz[0]//2, imgsz[1]//2]
    streamer = DataLoaderVideo(sources='0',imgsz=imgsz)
    LOGGER.info(f"Streamer loaded")
    vis = Visualizer('Joint - Images')
    cursor = Cursor()
    cv2.setMouseCallback('Joint - Images', cursor.set_cursor_position)
    with torch.inference_mode():
        for sources, img0, img, _ in streamer:
            img0 = img0[0]
            depth_map = depth_model.whole_inference(img, img_meta=None, rescale=True)
            position = cursor.position
            position = (position[0]-imgsz[1], position[1])
            distance = get_distance_at_point(depth_map.squeeze(), position[::-1])
            # Clear last line
            print("\033[A                             \033[A")
            print(f"Distance: {distance:.2f}m, average: {depth_map.mean():.2f}m, max: {depth_map.max():.2f}m, min: {depth_map.min():.2f}m")
            
            rendered = render_depth(depth_map.squeeze().cpu())
            pos = (cursor.x, cursor.y)
            vis.put_text(rendered, f"d: {distance:.2f}m", position, color=(0, 0, 255))
            vis.put_point(rendered, position, color=(0, 0, 255), radius=5)
            grid = grid_images([img0, rendered[...,::-1]],2)
            if not vis.update(grid):
                break

