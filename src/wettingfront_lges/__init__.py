"""Image analysis for LGES battery electrolyte filling experiment."""

import sys

if sys.version_info < (3, 10):
    from importlib_resources import files
else:
    from importlib.resources import files


__all__ = [
    "get_sample_path",
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
