"""Functions for reading instrument-generated data.

Everything in waltlabtools.read is automatically imported with
waltlabtools, so it can be accessed via, e.g.,

.. code-block:: python

   import waltlabtools as wlt  # waltlabtools main functionality

   my_hdx_report = wlt.read_hdx()  # extracts data from an HD-X file

"""

from inspect import signature
import os
from tkinter import filedialog

import pandas as pd

from .cal_curve import CalCurve
from .model import Model, models

__all__ = ["read_raw_hdx", "read_hdx"]


def _read_tsv(*args, **kwargs):
    """Reads a tsv file. See `pandas.read_csv` for more information."""
    kwargs["sep"] = "\t"
    return pd.read_csv(*args, **kwargs)


_PD_READERS = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "json": pd.read_json,
    "html": pd.read_html,
    "tsv": _read_tsv,
}
"""Mapping from file extensions to pandas read functions."""


def _get_file(filepath=None, **kwargs):
    """Returns filepath if provided, or asks the user to choose one.

    Parameters
    ----------
    filepath : str, path object or file-like object, optional
        The path to the file to import. Any valid string path is
        acceptable. The string could be a URL. Valid URL schemes include
        http, ftp, s3, gs, and file. Can also be any os.PathLike or any
        object with a `read()` method. If not provided, a
        `tkinter.filedialog` opens, prompting the user to select a file.
    **kwargs
        Other arguments are passed to the tkinter.filedialog.

    Returns
    -------
    io : str, path object or file-like object

    """
    if filepath is None:
        io = filedialog.askopenfilename(**kwargs)
    else:
        io = filepath
    return io


def read_raw_hdx(filepath=None, **kwargs) -> pd.DataFrame:
    """Read in an HD-X Sample Results Report or Run History.

    Essentially a wrapper for `pandas.read_csv` or `pandas.read_excel`.

    Parameters
    ----------
    filepath : str, path object or file-like object, optional
        The path to the file to import. Any valid string path is
        acceptable. The string could be a URL. Valid URL schemes include
        http, ftp, s3, gs, and file. Can also be any os.PathLike or any
        object with a `read()` method. If not provided, a
        `tkinter.filedialog` opens, prompting the user to select a file.

    Returns
    -------
    pandas.DataFrame

    See Also
    --------
    read.hdx : read in a spreadsheet and extract data automatically

    """
    io = _get_file(
        filepath,
        title="Choose a Sample Results Report or Run History File",
        filetypes=[
            ("Sample Results Report", "xls"),
            ("Run History", "csv"),
            ("All Files", "*"),
        ],
    )
    file_extension = os.fspath(io).split(".")[-1]
    reader = _PD_READERS[file_extension]
    reader_kwargs = {"header": 5 if file_extension == "xls" else 0}
    reader_kwargs.update(
        {
            key: value
            for key, value in kwargs.items()
            if (key in signature(reader).parameters)
            and (key not in signature(pd.DataFrame.pivot_table).parameters)
        }
    )
    return reader(io, **reader_kwargs)


def read_hdx(
    filepath=None,
    cal_curve=None,
    x_col: str = "Replicate Conc.",
    y_col: str = "Replicate AEB",
    index="Sample Barcode",
    columns=None,
    calibrators: tuple = ("Sample Type", "Calibrator"),
    samples: tuple = ("Sample Type", "Specimen"),
    sort: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Extracts data from an HD-X Sample Results Report or Run History.

    Transforms a spreadsheet into a pandas DataFrame whose columns are
    different assays/plexes (often corresponding to individual
    biomarkers) and whose rows are different samples. By default, the
    concentrations calculated by the HD-X software are used, but they
    can also be calculated independently from AEBs by passing a CalCurve
    object or a Model from which to generate a calibration curve.

    Parameters
    ----------
    filepath : str, path object, file-like object, or pandas.DataFrame
    optional
        The path to the file to import. Any valid string path is
        acceptable. The string could be a URL. Valid URL schemes include
        http, ftp, s3, gs, and file. Can also be any os.PathLike or any
        object with a `read()` method. Can also be a pandas.DataFrame
        if the data have already been imported. If `filepath` is not
        provided, a `tkinter.filedialog` opens, prompting the user to
        select a file.
    cal_curve : CalCurve, callable, Model, or str, optional
        To calculate concentrations from AEBs, pass one of the following
        types of arguments:

            - CalCurve: Calculate the concentrations using the
              CalCurve.inverse method.

            - callable: Transform data to concentrations with the
              given function.

            - Model: Generate a calibration curve from the data using
              the given model, and calculate concentrations using this
              calibration curve.

            - str: Should be an element of `models`. Generate a
              calibration curve from the data using the model named, and
              calculate concentrations using this calibration curve.

    x_col : str, default "Replicate Conc."
        Name of the column in the imported file to be used as the
        concentration. Ignored when `cal_curve` is a CalCurve object
        or callable.
    y_col : str, default "Replicate AEB"
        Name of the column in the imported file to be used as the
        signal (e.g., AEB), from which the concentration is calculated.
        Ignored unless `cal_curve` is provided. To use `cal_curve` to
        transform the concentrations rather than the AEBs, explicitly
        pass ``y_col="Replicate Conc."``.
    index: str or list of str, default "Sample Barcode"
        Column(s) of the spreadsheet to use as the index of the table,
        i.e., the unique barcodes for each sample. For example, to use
        plate well positions instead, pass ``index="Location"``.
    columns: str or list of str, optional
        Column(s) of the spreadsheet to use as the columns of the table
        uniquely specifying each biomarker/assay/plex. Default (None)
        is equivalent to passing ``["Assay", "Plex"]``.
    calibrators : tuple, default ("Sample Type", "Calibrator")
        Two-tuple of (colname, value) specifying the calibrators. For
        example, by default, all rows that have a "Sample Type" of
        "Calibrator" are counted as calibrators.
    samples : tuple, default ("Sample Type", "Specimen")
        Two-tuple of (colname, value) specifying the samples. For
        example, by default, all rows that have a "Sample Type" of
        "Specimen" are counted as samples and returned in the table.

    Returns
    -------
    pandas.DataFrame
        DataFrame whose rows (specified by `index`) are samples and
        whose columns are biomarkers/assays/plexes (specified by
        `columns`).

    See Also
    --------
    read.raw_hdx : read in a spreadsheet without transforming

    """
    # Import file.
    if isinstance(filepath, pd.DataFrame):
        raw_df = filepath.copy()
    else:
        raw_df = read_raw_hdx(filepath, **kwargs)

    # Form pivot table.
    pivot_table_kwargs = {
        "values": x_col,
        "index": index,
        "columns": ["Assay", "Plex"] if columns is None else columns,
        "sort": sort,
    }
    pivot_table_kwargs.update(
        {
            key: value
            for key, value in kwargs.items()
            if key in signature(pd.DataFrame.pivot_table).parameters
        }
    )
    return raw_df[raw_df[samples[0]] == samples[1]].pivot_table(**pivot_table_kwargs)
