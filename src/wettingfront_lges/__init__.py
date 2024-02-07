"""Image analysis for LGES battery electrolyte filling experiment."""

import sys

from .anode import analyze_anode
from .separator import Separator

if sys.version_info < (3, 10):
    from importlib_resources import files
else:
    from importlib.resources import files


__all__ = [
    "get_sample_path",
    "Separator",
]


def get_sample_path(*paths: str) -> str:
    """Get path to sample file.

    Arguments:
        paths: Subpaths under ``wettingfront_lges/samples/`` directory.

    Returns:
        Absolute path to the sample file.

    Examples:
        >>> get_sample_path() # doctest: +SKIP
        'path/wettingfront_lges/samples'
        >>> get_sample_path("myfile") # doctest: +SKIP
        'path/wettingfront_lges/samples/myfile'
    """
    return str(files("wettingfront_lges").joinpath("samples", *paths))


def anode_analyzer(k, v):
    """Registered as: ``Anode``.

    Entry of the configuration file must have ``parameters`` field, which contains the
    following sub-fields:

    - **path**: `str`
    - **sigma**: `number`
    - **fps**: `number` (optional)
    - **visual_output**: `str` (optional)
    - **data_output**: `str` (optional)
    - **plot_output**: `str` (optional)

    Refer to :func:`~.anode.analyze_anode` for more information.

    The following is the example for an YAML entry:

    .. code-block:: yaml

        foo:
            type: Anode
            parameters:
                path: foo.mp4
                sigma: 1
                data_output: output/foo.csv
    """
    analyze_anode(**v["parameters"], name=k)
