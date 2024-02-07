"""Electrolyte wetting front on anode.

Because of dense specks on the anode surface, wetting front is barely visible on
individual image. Instead, the difference between images is analyzed to detect the
movement of wetting front.
"""

# Two possible problems:
# 1. Memory issue if frames are too many.
# 2. Detection failure if wetting front does not move
# Solution: recursive prediction (e.g., Kalman filter)

import csv
import os

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm  # type: ignore
from scipy.ndimage import gaussian_filter  # type: ignore[import]

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
    for frame in iio.imiter(path, plugin="pyav"):
        gray = np.dot(frame, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        mean = np.mean(gray, axis=1)
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
    out_vid: str = "",
    out_data: str = "",
    out_plot: str = "",
    name: str = "",
):
    """Analyze the anode wetting video and save the result.

    Wetting height is normalized by the height of the image, i.e., ``0`` indicates no
    wetting and ``1`` indicates complete wetting.

    Arguments:
        path: Path to target video file.
        y_sigma: Sigma value for spatial Gaussian smoothing.
        t_sigma: Sigma value for temporal Gaussian smoothing.
        out_vid: Path to the output video file.
        out_data: Path to the output CSV file.
        out_plot: Path to the output plot file.
        name: Name of the analysis displayed on the progress bar.
    """

    def makedir(path):
        path = os.path.expandvars(path)
        dirname, _ = os.path.split(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        return path

    out_vid = makedir(out_vid)
    out_data = makedir(out_data)
    out_plot = makedir(out_plot)

    immeta = iio.immeta(path, plugin="pyav")
    fps = immeta["fps"]
    heights = []

    data = read_data(path)
    b = detect_wettingfront(data, y_sigma, t_sigma)

    if out_vid:
        codec = immeta["codec"]
        with iio.imopen(out_vid, "w", plugin="pyav") as out:
            out.init_video_stream(codec, fps=fps)
            for frame, h in tqdm.tqdm(
                zip(iio.imiter(path, plugin="pyav"), b),
                total=int(fps * immeta["duration"]),
                desc=name,
            ):
                H = frame.shape[0]
                frame[h, :] = (255, 0, 0)
                out.write_frame(frame)
                heights.append(1 - h / H)
    elif out_data:
        for frame, h in tqdm.tqdm(
            zip(iio.imiter(path, plugin="pyav"), b),
            total=int(fps * immeta["duration"]),
            desc=name,
        ):
            H = frame.shape[0]
            frame[h, :] = (255, 0, 0)
            heights.append(1 - h / H)

    if out_data or out_plot:
        times = np.arange(len(heights)) / fps

    if out_data:
        with open(out_data, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time (s)", "Wetting height"])
            for t, h in zip(times, heights):
                writer.writerow([t, h])

    if out_plot:
        fig, ax = plt.subplots()
        ax.plot(times, heights)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Wetting height [ratio]")
        fig.savefig(out_plot)
