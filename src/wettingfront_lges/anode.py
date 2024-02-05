"""Electrolyte wetting front on anode.

Because of dense specks on the anode surface, wetting front is barely visible on
individual image. Instead, the difference between images is analyzed to detect the
movement of wetting front.
"""

# Two possible problems:
# 1. Memory issue if frames are too many.
# 2. Detection failure if wetting front does not move
# Solution: recursive prediction (e.g., Kalman filter)

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm  # type: ignore
from scipy.ndimage import gaussian_filter  # type: ignore[import]

from fillyte.wettingfront.readers import fps as get_fps
from fillyte.wettingfront.readers import frame_count, frame_generator, frame_shape
from fillyte.wettingfront.writers import CSVWriter, ImageWriter

__all__ = [
    "read_data",
    "detect_wettingfront",
    "analyze_anode",
]


def read_data(path: str) -> npt.NDArray[np.float64]:
    """Read images from *path* and convert to consecutive wetting data.

    Each image from *path* is converted to grayscale, averaged along its
    1st axis, and sequentially appended to the return value.

    Arguments:
        path: Path to a visual media file containing target images.
            Can be video file, image file, or their glob pattern.
            Multipage image file is supported.

    Returns:
        Wetting data in 2-dimensional array.
            The 0th axis represents the frame number, and the 1st axis the
            pixel intensities of each frame averaged onto its y-axis.

    Examples:
        .. plot::
            :include-source:
            :context: reset

            >>> from wettingfront_lges import get_sample_path
            >>> from wettingfront_lges.anode import read_data
            >>> data = read_data(get_sample_path("anode.mp4"))
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(data) #doctest: +SKIP
    """
    ret = []
    for frame in frame_generator(path):
        mean = np.mean(frame, axis=1)
        ret.append(mean)
    return np.array(ret)


def detect_wettingfront(
    data: npt.NDArray, y_sigma: float, t_sigma: float
) -> npt.NDArray[np.int64]:
    """Detect wetting front from the wetting data.

    Arguments:
        data: Wetting data in 2-dimensional array.
            See :func:`read_data` for example.
        y_sigma: Sigma values for spatial Gaussian smoothing.
        t_sigma: Sigma values for temporal Gaussian smoothing.

    Returns:
        Y-coordinates of wetting fronts from consecutive data.

    Examples:
        .. plot::
            :include-source:
            :context: reset

            >>> from wettingfront_lges import get_sample_path
            >>> from wettingfront_lges.anode import read_data, detect_wettingfront
            >>> data = read_data(get_sample_path("anode.mp4"))
            >>> b = detect_wettingfront(data, 1, 1)
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.plot(b); plt.xlabel("frame #") #doctest: +SKIP
    """
    diff = gaussian_filter(data, (t_sigma, y_sigma), order=(1, 1), axes=(0, 1))
    return np.argmin(diff, axis=1)


def analyze_anode(
    path: str,
    y_sigma: float,
    t_sigma: float,
    *,
    fps: float = 0.0,
    visual_output: str = "",
    data_output: str = "",
    plot_output: str = "",
    name: str = "",
):
    """Analyze the anode wetting images and save the result.

    Wetting height is normalized by the height of the image, i.e., ``0`` indicates no
    wetting and ``1`` indicates complete wetting.

    Arguments:
        path: Path to a visual media file containing target images.
            Can be video file, image file, or their glob pattern.
            Multipage image file is supported.
        y_sigma: Sigma value for spatial Gaussian smoothing.
        t_sigma: Sigma value for temporal Gaussian smoothing.
        fps: FPS of the analysis output.
            If non-zero *fps* is passed, it is used to analyze and save the result.
            Else, :func:`fps` attempts to get the FPS from *path*.
        visual_output: Media path where the visual output will be saved.
        data_output: CSV file path where the data output will be saved.
        plot_output: Image file path where the data plot will be saved.
        name: Name of the analysis displayed on the progress bar.

    Notes:
        *visual_output* can be video file or image file. Formattable image path is
        supported to save each frame as separate file. If non-formattable GIF path is
        passed, frames are saved as multi-page image. Other multi-page image formats are
        not supported; if non-formattable, non-GIF path if passed, each frame overwrites
        the previous frame.
    """
    data = read_data(path)
    H, W = frame_shape(path)
    b = detect_wettingfront(data, y_sigma, t_sigma)

    def makedir(path):
        dirname, _ = os.path.split(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    if fps == 0.0:
        FPS = get_fps(path)
    else:
        FPS = fps

    if visual_output:
        fgen = frame_generator(path)
        makedir(visual_output)
        imgwriter = ImageWriter(
            visual_output,
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
            FPS,
        )
        next(imgwriter)
    if data_output:
        makedir(data_output)
        datawriter = CSVWriter(data_output)
        next(datawriter)
        HEADER = ["Wetting height"]
        if FPS != 0.0:
            HEADER.insert(0, "time (s)")
        datawriter.send(HEADER)
    if plot_output:
        makedir(plot_output)
        fig, ax = plt.subplots()

    for i in tqdm.tqdm(range(frame_count(path)), desc=name):
        y = b[i]
        if visual_output:
            image = cv2.cvtColor(next(fgen), cv2.COLOR_GRAY2RGB)
            cv2.line(image, (0, y), (W, y), (255, 0, 0), 1)
            imgwriter.send(image)  # type: ignore[arg-type]
        if data_output:
            DATA = [1 - y / H]
            if FPS != 0.0:
                DATA.insert(0, i / FPS)
            datawriter.send(DATA)
    if visual_output:
        imgwriter.close()
    if data_output:
        datawriter.close()
    if plot_output:
        ax.plot(1 - b / H)
        ax.set_xlabel("Frame #")
        ax.set_ylabel("Wetting height [ratio]")
        fig.savefig(plot_output)
