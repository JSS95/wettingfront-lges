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
from importlib.metadata import entry_points

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm  # type: ignore
import yaml
from scipy.ndimage import gaussian_filter  # type: ignore[import]

__all__ = [
    "x_means",
    "boundaries",
]


def x_means(path):
    """Yield 1-D array of averaged x values from video frames.

    Arguments:
        path: Path to video file.

    Yields:
        Averaged x vales.

    Examples:
        .. plot::
            :include-source:
            :context: reset

            >>> from wettingfront_lges import get_sample_path
            >>> from wettingfront_lges.anode import x_means
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(list(x_means(get_sample_path("anode.mp4")))) #doctest: +SKIP
    """
    for frame in iio.imiter(path, plugin="pyav"):
        gray = np.dot(frame, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        yield np.mean(gray, axis=1)


def boundaries(x_means: npt.ArrayLike, sigma_y: float, sigma_t: float) -> npt.NDArray:
    """Detect wetting front boundaries from :func:`x_means`.

    Arguments:
        x_means: Averaged x values.
        sigma_y: Sigma value for spatial Gaussian smoothing.
        sigma_t: Sigma value for tempral Gaussian smoothing.

    Returns:
        Y-coordinates of wetting front boundaries.

    Examples:
        .. plot::
            :include-source:
            :context: reset

            >>> from wettingfront_lges import get_sample_path
            >>> from wettingfront_lges.anode import x_means, boundaries
            >>> xm = list(x_means(get_sample_path("anode.mp4")))
            >>> bd = boundaries(xm, 1, 1)
            >>> import matplotlib.pyplot as plt #doctest: +SKIP
            >>> plt.imshow(xm); plt.plot(bd, np.arange(len(bd))) #doctest: +SKIP
    """
    diff = gaussian_filter(x_means, (sigma_t, sigma_y), order=(1, 1), axes=(0, 1))
    return np.argmin(diff, axis=1)


def anode_analyzer(name, fields):
    """Image analysis for unidirectional electrolyte imbibition in anode.

    The analyzer defines the following fields in configuration entry:

    - **model** (`str`, optional): Wetting front model, implemented by plugins.
    - **path** (`str`): Path to target video file.
    - **parameters**
        - **sigma_y** (`number`): Sigma value for spatial Gaussian smoothing.
        - **sigma_t** (`number`): Sigma value for temporal Gaussian smoothing.
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
            type: Anode
            path: foo.mp4
            parameters:
                sigma_y: 1
                sigma_t: 2
                fov_height: 4
            output:
                data: output/foo.csv
    """
    model = fields.get("model", None)
    if model is not None:
        MODELS = {}
        for ep in entry_points(group="wettingfront.models"):
            MODELS[ep.name] = ep
        model = MODELS[model].load()

    path = os.path.expandvars(fields["path"])

    sigma_y = fields["parameters"]["sigma_y"]
    sigma_t = fields["parameters"]["sigma_t"]
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

    immeta = iio.immeta(path, plugin="pyav")
    fps = immeta["fps"]

    xm = []
    for mean in tqdm.tqdm(
        x_means(path),
        total=int(fps * immeta["duration"]),
        desc=name + " (read)",
    ):
        xm.append(mean)
    xm = np.array(xm)
    bds = boundaries(xm, sigma_y, sigma_t)

    _, H = xm.shape
    if first_is_base:
        base = bds[0]
    else:
        base = H
    heights = (base - bds) / H * fov_height

    if output_model or output_data or output_plot:
        times = np.arange(len(heights)) / fps
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
            ax.set_ylabel("Height (mm)")
            ax.legend()
            fig.savefig(output_plot)

    if output_vid:
        codec = immeta["codec"]
        with iio.imopen(output_vid, "w", plugin="pyav") as out:
            out.init_video_stream(codec, fps=fps)
            for frame, b in tqdm.tqdm(
                zip(iio.imiter(path, plugin="pyav"), bds),
                total=int(fps * immeta["duration"]),
                desc=name + " (write)",
            ):
                frame[b, :] = (255, 0, 0)
                if 0 < base and base < H:
                    frame[base, :] = (0, 0, 255)
                out.write_frame(frame)
