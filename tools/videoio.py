"""Locate the full-resolution copy of a working video (see prepare_video.py)."""

from pathlib import Path

import cv2


def full_resolution(video, explicit=None):
    """(path, scale): the video to cut crops from and the factor from pixels of
    `video` (in which detections are given) to pixels of that video.

    For <name>_work.mp4 the sibling <name>_full.mp4 is used if it exists.
    """
    video = Path(video)
    path = Path(explicit) if explicit else None
    if path is None and video.stem.endswith("_work"):
        sibling = video.with_name(video.stem[:-len("_work")] + "_full.mp4")
        if sibling.exists():
            path = sibling
    if path is None:
        return video, 1.0
    width = lambda p: cv2.VideoCapture(str(p)).get(cv2.CAP_PROP_FRAME_WIDTH) or 1
    return path, width(path) / width(video)
