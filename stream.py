from data.loaders import LoadStreams
from utils.visualizers import Visualizer, grid_images
from utils.depth import render_depth
from models.dinov2_loader import *
import cv2
import numpy as np

if __name__=="__main__":
    streamer = LoadStreams(sources='0',imgsz=(480,640))
    vis = Visualizer('Joint - Images')
    vis_inp = Visualizer('Input')
    loader = DinoV2Loader("small", "dpt", "nyu")
    depth_model = loader.load_model()
    with torch.inference_mode():
        for sources, img0, img, _ in streamer:
            img0 = img0[0]
            img_np = np.ascontiguousarray(img.squeeze(0).permute(1,2,0).cpu().numpy()[...,::-1])
            print(f"Mean : {img_np.mean()}")
            vis_inp.update(img_np)
            depth_map = depth_model.whole_inference(img, img_meta=None, rescale=False)
            rendered = render_depth(depth_map.squeeze().cpu())
            grid = grid_images([img0, rendered[...,::-1]],2)
            if not vis.update(grid):
                break

