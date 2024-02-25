"""Jelly roll analysis."""

import os
from importlib.metadata import entry_points
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import curve_fit  # type: ignore[import-untyped]


def fit_jrwtf(t, x) -> tuple[Callable, tuple[np.float64]]:
    r"""Fit data to Washburn's equation [#f1]_ with y-axis offset.

    The data are fitted to:

    .. math::

        x = k \sqrt{ta} + b

    where :math:`k` is penetrativity of the liquid and :math:`b` is offset to
    compensate entrance length.

    Arguments:
        t (array_like, shape (M,)): Time.
        x (array_like, shape (M,)): Penetration length.

    Returns:
        func
            Washburn equation function f(t).
        (k, b)
            Fitted parameters.

    .. [#f1] Washburn, E. W. (1921). The dynamics of capillary flow.
             Physical review, 17(3), 273.
    """

    def func(t, k, b):
        return k * np.sqrt(t) + b

    ret, _ = curve_fit(func, t, x)
    return lambda t: func(t, *ret), ret


def jrwtf_analyzer(name, fields):
    """Jelly roll wetting front visualization.

    The analyzer defines the following fields in configuration entry:

    - **model** (`str`, optional): Wetting front model, implemented by plugins.
    - **data** (`str`): CSV file containing wetting front data.
        The CSV file must contain the following fields:
        - **ImbibitionStart** (`str`): Starting time in ISO8601 format.
        - **ImbibitionEnd** (`str`): Ending time in ISO8601 format.
        - **WettingHeight** (`number`): Wetting height in milimeter.
    - **output**:
        - **model** (`str`, optional): Path to the output YAML file.
            The model file stores model parameters.
        - **data** (`str`, optional): Path to the output CSV file.
            The data file stores wetting front data.
        - **plot** (`str`, optional): Path to the output plot file.
            The plot file visualizes wetting front data.

    The following is an example for an YAML entry:

    .. code-block:: yaml

        foo:
            type: JellyRollWettingFront
            model: Washburn_jellyroll
            data: foo.csv
            output:
                model: output/model.yml
                data: output/data.csv
                plot: output/plot.jpg
    """
    model = fields.get("model", None)
    if model is not None:
        MODELS = {}
        for ep in entry_points(group="wettingfront.models"):
            MODELS[ep.name] = ep
        model = MODELS[model].load()

    data = os.path.expandvars(fields["data"])

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

    df = pd.read_csv(data)
    start, end = map(pd.to_datetime, (df["ImbibitionStart"], df["ImbibitionEnd"]))
    dt = (end - start).dt.total_seconds()

    idx = dt.argsort()
    time, height = dt[idx], df["WettingHeight"][idx]

    if output_model:
        if model is None:
            params = ()
        else:
            func, params = model(time, height)
            predict = func(time)
        with open(output_model, "w") as f:
            yaml.dump(list(float(p) for p in params), f)

    if output_data:
        if model is None:
            pd.DataFrame({"time (s)": time, "height (mm)": height}).to_csv(
                output_data, index=False
            )
        else:
            pd.DataFrame(
                {"time (s)": time, "height (mm)": height, "fitted height (mm)": predict}
            ).to_csv(output_data, index=False)

    if output_plot:
        fig, ax = plt.subplots()
        ax.plot(time, height, label="data")
        if model is not None:
            ax.plot(time, predict, label="model")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Height (mm)")
        ax.legend()
        fig.savefig(output_plot)
