"""File readers."""

import glob
import mimetypes
import os
from typing import Generator, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageSequence

__all__ = [
    "frame_generator",
    "frame_shape",
    "frame_count",
    "fps",
]


def frame_generator(path: str) -> Generator[npt.NDArray[np.uint8], None, None]:
    """Yield grayscale frames from *path*.

    Arguments:
        path: Path to visual media file(s).

    Note:
        Path can be a single video file or a single image file. Multi-page image
        is supported. Plus, a glob pattern for these files is allowed.
    """
    files = glob.glob(os.path.expandvars(path))
    for f in files:
        mtype, _ = mimetypes.guess_type(f)
        if mtype is None:
            continue
        mtype, _ = mtype.split("/")
        if mtype == "image":
            with PIL.Image.open(f) as img:
                for frame in PIL.ImageSequence.Iterator(img):
                    yield np.array(frame.convert("L"))
        elif mtype == "video":
            cap = cv2.VideoCapture(f)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue


def frame_shape(path: str) -> Tuple[int, int]:
    """Return the shape of the frame from *path*, ``(-1, -1)`` if invalid.

    Frame shape is in ``(height, width)``.
    Only the first file from *path* is inspected. The other files are ignored.

    Arguments:
        path: Path to visual media file(s).

    Note:
        Path can be a single video file or a single image file. Multi-page image
        is supported, where the duration of each frame is converted to FPS.
        Plus, a glob pattern for these files is allowed.
    """
    path = glob.glob(os.path.expandvars(path))[0]
    mtype, _ = mimetypes.guess_type(path)
    if mtype is None:
        return (-1, -1)
    mtype, _ = mtype.split("/")
    if mtype == "image":
        with PIL.Image.open(path) as img:
            w, h = img.size
        return (h, w)
    elif mtype == "video":
        cap = cv2.VideoCapture(path)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        return (h, w)
    return (-1, -1)


def frame_count(path: str) -> int:
    """Return number of frames from *path*.

    Arguments:
        path: Path to visual media file(s).

    Note:
        Path can be a single video file or a single image file. Multi-page image
        is supported. Plus, a glob pattern for these files is allowed.
    """
    i = 0
    files = glob.glob(os.path.expandvars(path))
    for f in files:
        mtype, _ = mimetypes.guess_type(f)
        if mtype is None:
            continue
        mtype, _ = mtype.split("/")
        if mtype == "image":
            with PIL.Image.open(f) as img:
                i += getattr(img, "n_frames", 1)
        elif mtype == "video":
            cap = cv2.VideoCapture(f)
            i += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            continue
    return i


def fps(path: str) -> float:
    """Return FPS from *path*, ``0.0`` if invalid.

    Only the first file from *path* is inspected. The other files are ignored.

    Arguments:
        path: Path to visual media file(s).

    Note:
        Path can be a single video file or a single image file. Multi-page image
        is supported, where the duration of each frame is converted to FPS.
        Plus, a glob pattern for these files is allowed.
    """
    path = glob.glob(os.path.expandvars(path))[0]
    mtype, _ = mimetypes.guess_type(path)
    if mtype is None:
        return 0.0
    mtype, _ = mtype.split("/")
    if mtype == "image":
        with PIL.Image.open(path) as img:
            duration = img.info.get("duration")
        if duration is not None:
            return float(1000 / duration)
    elif mtype == "video":
        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps
    return 0.0
