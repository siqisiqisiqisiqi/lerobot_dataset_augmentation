import os
import json
import glob
from pathlib import Path

import cv2
import numpy as np


def load_video(video_path):
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
        try:
            # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
            video_frames_for_vis.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
            print(
                f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
                f"falling back to lexicographic sort."
            )
            video_frames_for_vis.sort()
    return video_frames_for_vis

def visualize_mask(image, mask):

    color = np.zeros_like(image)
    color[:, :, 1] = 255

    mask = mask.astype(bool)

    overlay = image.copy()
    overlay[mask] = overlay[mask] * 0.5 + color[mask] * 0.5

    return overlay.astype(np.uint8)


import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # set before importing pyplot
import matplotlib.pyplot as plt


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")

class VideoPointAnnotator:
    def __init__(self, video_path, frame_index=0):
        self.video_path = str(video_path)
        self.frame_index = frame_index

        # original frame
        self.image_bgr = None
        self.image_rgb = None

        # label=1: positive, label=0: negative
        self.points = []
        self.labels = []

        # current click mode
        self.mode = 1  # default: positive

        # matplotlib objects
        self.fig = None
        self.ax = None

        # for optional scatter artists
        self.point_artists = []

    def load_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.frame_index < 0 or self.frame_index >= total_frames:
            cap.release()
            raise ValueError(f"frame_index out of range: 0 ~ {total_frames - 1}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame {self.frame_index}")

        self.image_bgr = frame
        self.image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        print(f"Loaded frame {self.frame_index} / {total_frames - 1}")

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.image_rgb)
        self.ax.axis("off")

        mode_text = "POSITIVE" if self.mode == 1 else "NEGATIVE"

        title_text = (
            f"Mode: {mode_text}\n"
            "Left click: add point | p: positive | n: negative | "
            "u: undo | c: clear | d: done"
        )
        self.ax.set_title(title_text, fontsize=12)

        for (x, y), label in zip(self.points, self.labels):
            color = "lime" if label == 1 else "red"

            # filled center
            self.ax.plot(x, y, marker="o", markersize=4, color=color)

            # outer ring
            self.ax.plot(
                x, y,
                marker="o",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.2
            )

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:   # only left click
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        self.points.append((x, y))
        self.labels.append(self.mode)

        self.redraw()

    def on_key(self, event):
        if event.key == "p":
            self.mode = 1
            self.redraw()

        elif event.key == "n":
            self.mode = 0
            self.redraw()

        elif event.key == "u":
            if len(self.points) > 0:
                self.points.pop()
                self.labels.pop()
            self.redraw()

        elif event.key == "c":
            self.points = []
            self.labels = []
            self.redraw()

        elif event.key == "d":
            plt.close(self.fig)

    def run(self):
        self.load_frame()

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.redraw()
        plt.show()

        return np.array(self.points, dtype=np.int32), np.array(self.labels, dtype=np.int32)
    
if __name__ == "__main__":
    video_path = "./episode_000060.mp4"
    frame_index = 20

    annotator = VideoPointAnnotator(video_path, frame_index)
    points, labels = annotator.run()
    print(points)
    print(labels)