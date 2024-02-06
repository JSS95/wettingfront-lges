"""File writers."""

import csv
import mimetypes
from pathlib import Path
from typing import Generator, List

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image
from matplotlib.animation import AbstractMovieWriter, writers

__all__ = [
    "MplImagesWriter",
    "ImageWriter",
    "CSVWriter",
]


@writers.register("images")
class MplImagesWriter(AbstractMovieWriter):
    """Class to save matplotlib artists as image files.

    An instance of this class can be given as the ``writer`` argument of
    :meth:`matplotlib.animation.Animation.save`. Its constructer is registered as
    ``"images"``.
    """

    @classmethod
    def isAvailable(cls):
        """Return whether the class is available for writing."""
        return True

    def setup(self, fig, outfile, dpi=None):
        """Setup for writing the image files.

        Arguments:
            fig: Figure to grab the rendered frames from.
            outfile: Filename of the resulting image files.
                Filename can contain string formats, e.g. ``img%03d.jpg``.
            dpi: DPI of the output file. Default: ``fig.dpi``.
                This, with the figure size, controls the size in pixels of the
                resulting image files.
        """
        Path(outfile).parent.resolve(strict=True)
        self.outfile = outfile
        try:
            outfile % 0
            self.formattable = True
        except TypeError:
            self.formattable = False
        self.fig = fig
        if dpi is None:
            dpi = self.fig.dpi
        self.dpi = dpi
        self._frame_counter = 0

    def grab_frame(self, **savefig_kwargs):
        """Grab the image information from the figure and save as an image.

        Keyword arguments in *savefig_kwargs* are passed on to the
        :meth:`matplotlib.figure.Figure.savefig` call that saves the figure except:

        - *dpi*, *bbox_inches*: These may not be passed because each frame of the
           animation much be exactly the same size in pixels.
        """
        for k in ("dpi", "bbox_inches", "format"):
            if k in savefig_kwargs:
                raise TypeError(f"grab_frame got an unexpected keyword argument {k!r}")
        if self.formattable:
            path = self.outfile % self._frame_counter
        else:
            path = self.outfile
        with open(path, "wb") as file:
            self.fig.savefig(file, dpi=self.dpi, **savefig_kwargs)
        self._frame_counter += 1

    def finish(self):
        """Finish any processing for writing the images."""
        pass


def ImageWriter(
    path: str, fourcc: int, fps: float
) -> Generator[None, npt.NDArray[np.uint8], None]:
    """Coroutine to write incoming RGB images into image file(s) or video file.

    This function supports several ways to write image data:

    #. Multiple single-page image files
        *path* is formattable path with image format (e.g., ``img%02d.jpg``).
    #. Single image file
        *path* is non-formattable path with image format (e.g., ``img.gif``).
        If the format supports multipage image, *fps* is used.
        If the format does not support multipage image, only the first image is written.
    #. Single video file
        *path* is non-formattable path with video format (e.g., ``img.mp4``).
        *fourcc* and *fps* is used to encode the video.

    Warning:
        When writing into a single image file, sending too many images will cause
        memory issue.

    Arguments:
        path: Resulting file path.
            Can have either image extension or video extension.
        fourcc: Result of :func:`cv2.VideoWriter_fourcc`.
            Specifies encoder to write the video. Ignored if *path* is not video.
        fps: Frame rate of incoming images.
            Specifies frame rate to write multipage image file or video file.
            Ignored if *path* is single-page image file(s).

    Note:
        Type of *path* (image vs video) is detected by :mod:`mimetypes`.
        Image file is written by :meth:`PIL`, and video file is written by
        :obj:`cv2.VideoWriter`.

    Examples:
        .. code-block:: python

            gen = ImageWriter(...)
            next(gen)  # Initialize the coroutine
            gen.send(img1)
            gen.send(img2)
            gen.close()  # Close the file
    """
    try:
        path % 0
        formattable = True
    except TypeError:
        formattable = False

    mtype, _ = mimetypes.guess_type(path)
    if mtype is None:
        raise TypeError(f"Invalid path: {path}.")
    ftype, _ = mtype.split("/")

    if ftype == "image":
        if formattable:
            i = 0
            while True:
                img = yield
                PIL.Image.fromarray(img).save(path % i)
                i += 1
        else:
            images = []
            img = yield
            try:
                while True:
                    images.append(PIL.Image.fromarray(img))
                    img = yield
            finally:
                if fps == 0.0:
                    try:
                        images[0].save(
                            path,
                            save_all=True,
                            append_images=images[1:],
                        )
                    except Exception:
                        images[-1].save(path)
                else:
                    try:
                        images[0].save(
                            path,
                            save_all=True,
                            append_images=images[1:],
                            duration=1000 / fps,
                        )
                    except Exception:
                        images[-1].save(path)
    elif ftype == "video":
        img = yield
        h, w = img.shape[:2]
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        try:
            while True:
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                img = yield
        finally:
            writer.release()
    else:
        raise TypeError(f"Unsupported mimetype: {mtype}.")


def CSVWriter(path: str) -> Generator[None, List, None]:
    """Coroutine to write incoming data to CSV file.

    Arguments:
        path: Resulting file path.

    Examples:
        .. code-block:: python

            gen = CSVWriter("result.csv")
            next(gen)  # Initialize the coroutine
            gen.send([1, 2, 3])
            gen.send(["foo", "bar", "baz"])
            gen.close()  # Close the file
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        while True:
            data = yield
            writer.writerow(data)
