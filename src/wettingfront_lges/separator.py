"""Electrolyte wetting front on separator.

Because separator image has clearly distinguishable regions, the boundary is directly
acquired from each image.
"""

import os
from typing import Any, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm  # type: ignore
from scipy.ndimage import gaussian_filter1d  # type: ignore[import]

from fillyte.wettingfront.cache import attrcache
from fillyte.wettingfront.readers import fps as get_fps
from fillyte.wettingfront.readers import frame_count, frame_generator
from fillyte.wettingfront.writers import CSVWriter, ImageWriter

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

            >>> import cv2
            >>> from wettingfront_lges import get_sample_path
            >>> from wettingfront_lges.separator import Separator
            >>> path = get_sample_path("separator.jpg")
            >>> img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            >>> sep = Separator(img, sigma=1)
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
        image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        _, W = self.image.shape
        h = self.boundary()
        cv2.line(image, (0, h), (W, h), (255, 0, 0), 1)
        return image  # type: ignore[return-value]


def analyze_separator(
    path: str,
    sigma: float,
    *,
    fps: float = 0.0,
    visual_output: str = "",
    data_output: str = "",
    plot_output: str = "",
    name: str = "",
):
    """Analyze the separator wetting images and save the result.

    Wetting height is normalized by the height of the image, i.e., ``0`` indicates no
    wetting and ``1`` indicates complete wetting.

    Arguments:
        path: Path to a visual media file containing target images.
            Can be video file, image file, or their glob pattern.
            Multipage image file is supported.
        sigma: Sigma value for spatial Gaussian smoothing.
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

    def makedir(path):
        dirname, _ = os.path.split(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    if fps == 0.0:
        FPS = get_fps(path)
    else:
        FPS = fps

    if visual_output:
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
        wettingheights = []
        fig, ax = plt.subplots()

    for i, frame in enumerate(
        tqdm.tqdm(frame_generator(path), total=frame_count(path), desc=name)
    ):
        sep = Separator(frame, sigma)
        H = frame.shape[0]
        if visual_output:
            imgwriter.send(sep.draw())
        if data_output:
            DATA: List[Any] = [sep.wetting_height() / H]
            if FPS != 0.0:
                DATA.insert(0, i / FPS)
            datawriter.send(DATA)
        if plot_output:
            wettingheights.append(sep.wetting_height() / H)

    if visual_output:
        imgwriter.close()
    if data_output:
        datawriter.close()
    if plot_output:
        ax.plot(wettingheights)
        ax.set_xlabel("Frame #")
        ax.set_ylabel("Wetting height [ratio]")
        fig.savefig(plot_output)
