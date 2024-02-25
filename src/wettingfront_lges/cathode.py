"""Electrolyte wetting front on cathode."""

import csv
import os
from importlib.metadata import entry_points

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm  # type: ignore
import yaml
from scipy.ndimage import gaussian_filter1d  # type: ignore[import]
from scipy.signal import find_peaks  # type: ignore[import]

from .cache import attrcache

__all__ = [
    "Cathode",
]


class Cathode:
    """Unidirectional wetting front on cathode.

    This class assumes that the wetting front is represented by a horizontal boundary.
    The boundary is detected by finding the abrupt change of row-wise averaged pixel
    intensities.

    Arguments:
        image: Grayscale target image.
        sigma: Sigma value for Gaussian filtering.
        base: Y coordinate of wetting base. ``None`` indicates bottom edge.
        peak_height: Required height of peaks to find wetting front.

    Examples:
        .. plot::
            :include-source:
            :context: reset

            >>> import numpy as np, imageio.v3 as iio
            >>> from wettingfront_lges import get_sample_path
            >>> from wettingfront_lges.cathode import Cathode
            >>> img = iio.imread(get_sample_path("cathode.jpg"))
            >>> gray = np.dot(img, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            >>> cat = Cathode(gray, sigma=1, base=182, peak_height=1.0)
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(cat.draw()) #doctest: +SKIP
    """

    def __init__(
        self,
        image: npt.NDArray[np.uint8],
        sigma: float,
        base: int | None = None,
        peak_height: float | None = None,
    ):
        """Initialize the instance.

        Set *image* to be immutable and process *base*.
        """
        self._image = image
        self._image.setflags(write=False)
        self._sigma = sigma
        self._base = image.shape[0] if base is None else base
        self._peak_height = peak_height

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
        """Wetting base."""
        return self._base

    @property
    def peak_height(self) -> float | None:
        """Required height for peak finding."""
        return self._peak_height

    @attrcache("_ydiff")
    def ydiff(self) -> npt.NDArray[np.float64]:
        """Gaussian derivative of row-wise averaged pixel intensities.

        Gaussian derivative is calculated using :attr:`self.sigma`.
        """
        xmean = np.mean(self.image, axis=1)
        return -gaussian_filter1d(xmean, self.sigma, order=1, mode="nearest")

    @attrcache("_wetting_front")
    def wetting_front(self) -> np.int64:
        """Y coordinate of wetting front.

        Wetting front is found by searching for the peak with the smallest Y
        coordinate in :meth:`ydiff`. Only the peaks higher than
        :meth:`peak_height` are considered.
        """
        peaks, _ = find_peaks(self.ydiff(), height=self.peak_height)
        return peaks[0]

    def wetting_height(self) -> np.int64:
        """Height of wetting."""
        return self.base - self.wetting_front()

    def draw(self) -> npt.NDArray[np.uint8]:
        """Return visualization result in RGB format."""
        image = np.repeat(self.image[..., np.newaxis], 3, axis=-1)
        image[self.wetting_front(), :] = (255, 0, 0)
        image[self.base, :] = (0, 255, 0)
        return image  # type: ignore[return-value]


def cathode_analyzer(name, fields):
    """Image analysis for unidirectional electrolyte imbibition in cathode.

    The analyzer defines the following fields in configuration entry:

    - **model** (`str`, optional): Wetting front model, implemented by plugins.
    - **path** (`str`): Path to target video file.
    - **parameters**
        - **fov_height** (`number`): Height of the field of view in milimeters.
        - **start** (`number`, optional): The point where analysis starts (in seconds).
            If not specified, analysis starts with the first frame in the video.
        - **sigma** (`number`): Parameter for :class:`Cathode`.
        - **base** (`int`, optional): Parameter for :class:`Cathode`.
        - **peak_height** (`number`, optional): Parameter for :class:`Cathode`.
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
            type: Cathode
            model: Washburn
            path: foo.mp4
            parameters:
                fov_height: 4
                start: 3
                sigma: 1
                base: 182
                peak_height: 1
            output:
                vid: output/vid.mp4
    """
    model = fields.get("model", None)
    if model is not None:
        MODELS = {}
        for ep in entry_points(group="wettingfront.models"):
            MODELS[ep.name] = ep
        model = MODELS[model].load()

    path = os.path.expandvars(fields["path"])

    fov_height = fields["parameters"]["fov_height"]
    start = fields["parameters"].get("start", 0)
    sigma = fields["parameters"]["sigma"]
    base = fields["parameters"].get("base")
    peak_height = fields["parameters"].get("peak_height")

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

    immeta = iio.immeta(path, plugin="pyav")
    fps = immeta["fps"]

    def yield_result(path):
        start_fnum = start * fps
        for i, frame in tqdm.tqdm(
            enumerate(iio.imiter(path, plugin="pyav")),
            total=int(fps * immeta["duration"]),
            desc=name,
        ):
            H = frame.shape[0]
            if i >= start_fnum:
                gray = np.dot(frame, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                cat = Cathode(gray, sigma, base, peak_height)
                yield cat.draw(), cat.wetting_height() / H * fov_height
            else:
                yield frame, None

    times = []
    heights = []
    if output_vid:
        codec = immeta["codec"]
        with iio.imopen(output_vid, "w", plugin="pyav") as out:
            out.init_video_stream(codec, fps=fps)
            for i, (frame, h) in enumerate(yield_result(path)):
                out.write_frame(frame)
                if h is not None:
                    times.append(i / fps)
                    heights.append(h)
    elif output_data or output_plot:
        for i, (frame, h) in enumerate(yield_result(path)):
            if h is not None:
                times.append(i / fps)
                heights.append(h)

    if output_model or output_data or output_plot:
        if model is None:
            params = ()
        else:
            func, params = model(times, heights)
            predict = func(times)

        if output_model:
            with open(output_model, "w") as f:
                yaml.dump(list(float(p) for p in params), f)

        if output_data:
            with open(output_data, "w", newline="") as f:
                writer = csv.writer(f)
                if model is None:
                    writer.writerow(["time (s)", "height (mm)"])
                    for t, h in zip(times, heights):
                        writer.writerow([t, h])
                else:
                    writer.writerow(["time (s)", "height (mm)", "fitted height (mm)"])
                    for t, h, p in zip(times, heights, predict):
                        writer.writerow([t, h, p])

        if output_plot:
            fig, ax = plt.subplots()
            ax.plot(times, heights, label="data")
            if model is not None:
                ax.plot(times, predict, label="model")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("height (mm)")
            ax.legend()
            fig.savefig(output_plot)
