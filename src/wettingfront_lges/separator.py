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
import yaml
from scipy.ndimage import gaussian_filter1d  # type: ignore[import]
from wettingfront import fit_washburn

from .cache import attrcache

__all__ = [
    "Separator",
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
        sigma: Sigma value for Gaussian filtering.
        base: Y coordinate of fluid meniscus.

    Examples:
        .. plot::
            :include-source:
            :context: reset

            >>> import numpy as np, imageio.v3 as iio
            >>> from wettingfront_lges import get_sample_path, Separator
            >>> img = iio.imread(get_sample_path("separator.jpg"))
            >>> gray = np.dot(img, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            >>> sep = Separator(gray, sigma=1, base=250)
            >>> sep.wetting_height()
            48
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(sep.draw()) #doctest: +SKIP
    """

    def __init__(
        self, image: npt.NDArray[np.uint8], sigma: float, base: int | None = None
    ):
        """Initialize the instance.

        *image* is set to be immutable.
        """
        self._image = image
        self._image.setflags(write=False)
        self._sigma = sigma
        self._base = base

    @property
    def image(self) -> npt.NDArray[np.uint8]:
        """Grayscale target image.

        Note:
            This array is immutable to allow caching.
        """
        return self._image

    @property
    def sigma(self) -> float:
        """Sigma value for Gaussian filtering.

        Kernel size is automatically determined from sigma.
        """
        return self._sigma

    @property
    def base(self) -> int:
        """Y coordinate of fluid meniscus.

        This value is used to determine the actual wetting height.
        ``None`` indicates the lower edge of image.
        """
        if self._base is None:
            return self.image.shape[0]
        return self._base

    def ydiff(self) -> npt.NDArray[np.float64]:
        """Difference of row-wise averaged pixel intensities.

        Values are smoothed using Gaussian filter with :attr:`self.sigma`.
        If sigma is zero, the data is not smoothed.
        """
        mean = np.mean(self.image, axis=1)
        if self.sigma == 0:
            ret = np.abs(np.diff(mean))
        else:
            ret = np.abs(gaussian_filter1d(mean, self.sigma, order=1))
        return ret

    @attrcache("_boundary")
    def boundary(self) -> np.int64:
        """Y coordinate where the boundary exists."""
        return np.argmax(self.ydiff())

    def wetting_height(self) -> np.int64:
        """Height of wetting."""
        return self.base - self.boundary()

    def draw(self) -> npt.NDArray[np.uint8]:
        """Return visualization result in RGB format."""
        image = np.repeat(self.image[..., np.newaxis], 3, axis=-1)
        h = self.boundary()
        image[h, :] = (255, 0, 0)
        if 0 < self.base and self.base < image.shape[0]:
            image[self.base, :] = (0, 0, 255)
        return image  # type: ignore[return-value]


def separator_analyzer(name, fields):
    """Image analysis for unidirectional electrolyte imbibition in separator.

    The analyzer defines the following fields in configuration entry:

    - **path** (`str`): Path to target video file.
    - **parameters**
        - **sigma** (`number`): Sigma value for spatial Gaussian smoothing.
        - **fov_height** (`number`): Height of the field of view in milimeters.
        - **first_is_base** (`bool`, optional): Whether the first frame's wetting front
            is baseline.
    - **output**:
        - **model** (`str`, optional): Path to the output YAML file.
            The model file stores model parameters.
        - **data** (`str`, optional): Path to the output CSV file.
            The data file stores wetting front data.
        - **plot** (`str`, optional): Path to the output plot file.
            The plot file visualizes wetting front data.
        - **vid** (`str`, optional): Path to the output video file.
            The video file shows the wetting front in the input video.

    The following is an example for an YAML entry:

    .. code-block:: yaml

        foo:
            type: Separator
            path: foo.mp4
            parameters:
                sigma: 1
                fov_height: 4
            output:
                data: output/foo.csv
    """
    path = os.path.expandvars(fields["path"])

    sigma = fields["parameters"]["sigma"]
    fov_height = fields["parameters"]["fov_height"]
    first_is_base = fields["parameters"].get("first_is_base", False)

    def makedir(path):
        path = os.path.expandvars(path)
        dirname, _ = os.path.split(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        return path

    output = fields.get("output", {})
    output_model = makedir(output.get("model", ""))
    output_data = makedir(output.get("data", ""))
    output_plot = makedir(output.get("plot", ""))
    output_vid = makedir(output.get("vid", ""))

    def yield_result(path):
        for i, frame in tqdm.tqdm(
            enumerate(iio.imiter(path, plugin="pyav")),
            total=int(fps * immeta["duration"]),
            desc=name,
        ):
            H = frame.shape[0]
            gray = np.dot(frame, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            if i == 0:
                sep = Separator(gray, sigma)
                base = sep.boundary()
            else:
                if first_is_base:
                    sep = Separator(gray, sigma, base)
                else:
                    sep = Separator(gray, sigma)
            yield sep.draw(), sep.wetting_height() / H * fov_height

    immeta = iio.immeta(path, plugin="pyav")
    fps = immeta["fps"]
    heights = []
    if output_vid:
        codec = immeta["codec"]
        with iio.imopen(output_vid, "w", plugin="pyav") as out:
            out.init_video_stream(codec, fps=fps)
            for frame, h in yield_result(path):
                out.write_frame(frame)
                heights.append(h)
    elif output_data or output_plot:
        for frame, h in yield_result(path):
            heights.append(h)

    if output_model or output_data or output_plot:
        times = np.arange(len(heights)) / fps
        k, a, b = fit_washburn(times, heights)
        washburn = k * np.sqrt(times - a) + b

        if output_model:
            with open(output_model, "w") as f:
                yaml.dump(dict(k=float(k), a=float(a), b=float(b)), f)

        if output_data:
            with open(output_data, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time (s)", "height (mm)", "fitted height (mm)"])
                for t, h, w in zip(times, heights, washburn):
                    writer.writerow([t, h, w])

        if output_plot:
            fig, ax = plt.subplots()
            ax.plot(times, heights, label="data")
            ax.plot(times, washburn, label="model")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("height (mm)")
            fig.savefig(output_plot)
