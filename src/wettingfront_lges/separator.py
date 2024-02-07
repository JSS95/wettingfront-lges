"""Electrolyte wetting front on separator.

Because separator image has clearly distinguishable regions, the boundary is directly
acquired from each image.
"""

import csv
import os

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm  # type: ignore
from scipy.ndimage import gaussian_filter1d  # type: ignore[import]

from .cache import attrcache

__all__ = [
    "Separator",
    "analyze_separator",
]


class Separator:
    """Wetting front on separator.

    This class assumes that the wetting front is represented by a horizontal boundary.
    The boundary is detected by finding the location where the row-wise averaged pixel
    intensities abruptly change.

    :meth:`wetting_height` returns the height of the wetting front. :meth:`draw` returns
    the visualization result.

    Arguments:
        image: Grayscale target image.
        sigma: Sigma value for Gaussian blurring.

    Examples:
        .. plot::
            :include-source:
            :context: reset

            >>> import numpy as np, imageio.v3 as iio
            >>> from wettingfront_lges import get_sample_path
            >>> from wettingfront_lges.separator import Separator
            >>> img = iio.imread(get_sample_path("separator.jpg"))
            >>> gray = np.dot(img, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            >>> sep = Separator(gray, sigma=1)
            >>> sep.wetting_height()
            54
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(sep.draw()) #doctest: +SKIP
    """

    def __init__(self, image: npt.NDArray[np.uint8], sigma: float):
        """Initialize the instance.

        *image* is set to be immutable.
        """
        self._image = image
        self._image.setflags(write=False)
        self._sigma = sigma

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """Grayscale target image.

        Note:
            This array is immutable to allow caching.
        """
        return self._image

    @property
    def sigma(self) -> float:
        """Sigma value for Gaussian blurring.

        Kernel size is automatically determined from sigma.
        """
        return self._sigma

    @attrcache("_ydiff")
    def ydiff(self) -> npt.NDArray[np.float64]:
        """Difference of row-wise averaged pixel intensities.

        Values are smoothed using Gaussian filter with :attr:`self.sigma`. If sigma is
        zero, the data is not smoothed.

        Note:
            The result is cached and must not be modified.
        """
        mean = np.mean(self.image, axis=1)
        if self.sigma == 0:
            ret = np.abs(np.diff(mean))
        else:
            ret = np.abs(gaussian_filter1d(mean, self.sigma, order=1))
        ret.setflags(write=False)
        return ret

    @attrcache("_boundary")
    def boundary(self) -> np.int64:
        """Y coordinate where the boundary exists."""
        return np.argmax(self.ydiff())

    def wetting_height(self) -> int:
        """Distance between :meth:`self.boundary` and the lower edge of the image."""
        H, _ = self.image.shape
        return int(H - self.boundary())

    def draw(self) -> npt.NDArray[np.uint8]:
        """Return visualization result in RGB format."""
        image = np.repeat(self.image[..., np.newaxis], 3, axis=-1)
        h = self.boundary()
        image[h, :] = (255, 0, 0)
        return image  # type: ignore[return-value]


def analyze_separator(
    path: str,
    sigma: float,
    *,
    out_vid: str = "",
    out_data: str = "",
    out_plot: str = "",
    name: str = "",
):
    """Analyze the separator wetting video and save the result.

    Wetting height is normalized by the height of the image, i.e., ``0`` indicates no
    wetting and ``1`` indicates complete wetting.

    Arguments:
        path: Path to target video file.
        sigma: Sigma value for spatial Gaussian smoothing.
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

    def yield_result(path):
        for frame in tqdm.tqdm(
            iio.imiter(path, plugin="pyav"),
            total=int(fps * immeta["duration"]),
            desc=name,
        ):
            H = frame.shape[0]
            gray = np.dot(frame, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            sep = Separator(gray, sigma)
            yield sep.draw(), sep.wetting_height() / H

    immeta = iio.immeta(path, plugin="pyav")
    fps = immeta["fps"]
    heights = []
    if out_vid:
        codec = immeta["codec"]
        with iio.imopen(out_vid, "w", plugin="pyav") as out:
            out.init_video_stream(codec, fps=fps)
            for frame, h in yield_result(path):
                out.write_frame(frame)
                heights.append(h)
    elif out_data or out_plot:
        for frame, h in yield_result(path):
            heights.append(h)

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
