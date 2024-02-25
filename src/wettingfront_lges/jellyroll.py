"""Jelly roll analysis."""

import os

import matplotlib.pyplot as plt
import pandas as pd


def jrwtf_analyzer(name, fields):
    """Jelly roll wetting front visualization.

    The analyzer defines the following fields in configuration entry:

    - **data** (`str`): CSV file containing wetting front data.
        The CSV file must contain the following fields:
        - **ImbibitionStart** (`str`): Starting time in ISO8601 format.
        - **ImbibitionEnd** (`str`): Ending time in ISO8601 format.
        - **WettingHeight** (`number`): Wetting height in milimeter.
    - **output**:
        - **data** (`str`, optional): Path to the output CSV file.
            The data file stores wetting front data.
        - **plot** (`str`, optional): Path to the output plot file.
            The plot file visualizes wetting front data.

    The following is an example for an YAML entry:

    .. code-block:: yaml

        foo:
            type: JellyRollWettingFront
            data: foo.csv
            output:
                data: output/data.csv
                plot: output/plot.jpg
    """
    data = os.path.expandvars(fields["data"])

    def makedir(path):
        path = os.path.expandvars(path)
        dirname, _ = os.path.split(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        return path

    output = fields.get("output", {})
    output_data = makedir(output.get("data", ""))
    output_plot = makedir(output.get("plot", ""))

    df = pd.read_csv(data)
    start, end = map(pd.to_datetime, (df["ImbibitionStart"], df["ImbibitionEnd"]))
    dt = (end - start).dt.total_seconds()

    idx = dt.argsort()
    time, height = dt[idx], df["WettingHeight"][idx]

    if output_data:
        pd.DataFrame({"time (s)": time, "height (mm)": height}).to_csv(
            output_data, index=False
        )

    if output_plot:
        fig, ax = plt.subplots()
        ax.plot(time, height, "x")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Height (mm)")
        fig.savefig(output_plot)
