"""Jelly roll analysis."""


def jrwtf_analyzer(name, fields):
    """Jelly roll wetting front visualization.

    The analyzer defines the following fields in configuration entry:

    - **data** (`str`): CSV file containing wetting front data.
    - **output**:
        - **plot** (`str`, optional): Path to the output plot file.
            The plot file visualizes wetting front data.

    The following is an example for an YAML entry:

    .. code-block:: yaml

        foo:
            type: JellyRollWettingFront
            data: foo.csv
            output:
                plot: output/plot.jpg
    """
    ...
