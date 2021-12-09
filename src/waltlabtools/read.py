"""Functions for reading in data from Quanterix instruments.

This module provides tools for interacting with a
`Quanterix Simoa HD-X Analyzer
<https://www.quanterix.com/instruments/simoa-hd-x-analyzer/>`__.

In addition to the dependencies for waltlabtools,
waltlabtools.read also requires pandas 0.25 or greater.

The public functions in waltlabtools.read can be accessed via,
e.g.,

.. code-block:: python

   import waltlabtools as wlt  # waltlabtools main functionality
   import waltlabtools.read  # for Quanterix data

   subset_data = wlt.read.quanterix()  # read run history

if also using other functionality from the waltlabtools package, or

.. code-block:: python

   from waltlabtools import read_quanterix  # for Quanterix data

   subset_data = read.quanterix()  # read run history

if using only the waltlabtools.readmodule.


-----


"""

import os
from tkinter import filedialog
import pandas as pd

__all__ = ["hdx"]


_hdx = {}

_hdx["files"] = {
    ("sample", "results"): {
        "filedialog": {
            "title": "Choose a Sample Results Report File",
            "filetypes": [("Excel 97–2004 Workbook", "xls")]},
        "args": {"header": 5}},
    ("run", "history"): {
        "filedialog": {
            "title": "Choose a Run History File",
            "filetypes": [("Comma-Separated Values", "csv")]},
        "args": {"header": 0}},
    ("any",): {
        "filedialog": {
            "title": "Choose a Sample Results Report or Run History File",
            "filetypes": [
                ("Excel 97–2004 Workbook", "xls"),
                ("Comma-Separated Values", "csv"),
                ("All Files", "*")]},
        "args": {}},
}


_hdx["export_types"] = {
    "xls": ("sample", "results"),
    "csv": ("run", "history")
}
_hdx["export_types"][("any",)] = list(set(_hdx["export_types"].values()))


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
        io = filedialog.askopenfilenames(**kwargs)
    else:
        io = filepath
    return io


_filetype_readers = {
    "csv": pd.read_csv,
    "excel": pd.read_excel,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "opendocument": pd.read_excel,
    "odf": pd.read_excel
}

_reader_args = {reader: () for reader in set(_filetype_readers.values())}
for reader in _reader_args:
    func = reader
    while hasattr(func, '__wrapped__'):
        func = func.__wrapped__
    _reader_args[reader] = set(func.__code__.co_varnames)


def _cols_dropped(raw_table: pd.DataFrame,
        drop_cols: str = "blank") -> pd.DataFrame:
    """Drops unimportant columns from a pandas DataFrame.

    [extended_summary]

    Parameters
    ----------
    raw_table : pd.DataFrame
        DataFrame without columns dropped.
    drop_cols : {"blank", "uniform", "keep"}, default "blank"
        Should any columns be automatically dropped from the input file?
        Options:

            - `"blank"` : Drop all columns that are blank.

            - `"uniform"` : Drop all columns that have the same
              value for all rows, including blank columns.

            - `"keep"` : Do not drop any columns.


    Returns
    -------
    pd.DataFrame
        [description]
    """
    if drop_cols == "keep":
        return raw_table
    elif drop_cols == "uninformative":
        uninformative_cols = [colname for colname in raw_table.columns
            if len(set(raw_table[colname])) <= 1]
        return raw_table.drop(columns=uninformative_cols)
    else:
        return raw_table.dropna(axis="columns", how="all")


def _get_file_extension(io) -> str:
    return os.fspath(io).rsplit('.', maxsplit=1)[-1]


def _hdx_export_type_tuple(export_type=None) -> tuple:
    export_lowercase = str(export_type).casefold()
    for export_tuple in _hdx["files"]:
        if all((word in export_lowercase) for word in export_tuple):
            return export_tuple
    return ("any",)


def _hdx_ext_tuple(io=None) -> tuple:
    extension = _get_file_extension(io)
    if extension in _hdx["export_types"]:
        if isinstance(_hdx["export_types"][extension], tuple):
            return _hdx["export_types"][extension]
        else:
            filepath_str = str(os.fspath(io))
            for export_tuple in _hdx["export_types"][extension]:
                if all((word in filepath_str) for word in export_tuple):
                    return export_tuple
    raise ValueError("Export type not recognized.")


def _read(io, reader, new_args, **kwargs) -> pd.DataFrame:
    kwargs_ = {key: value for key, value in kwargs.items()
        if key in _reader_args[reader]}
    kwargs_.update(new_args)
    return reader(io, **kwargs_)


def _read_hdx_table(io, export_tuple, **kwargs) -> pd.DataFrame:
    value_error = None
    extension = _hdx["files"][export_tuple]["filedialog"]["filetypes"][0][1]
    new_args = _hdx["files"][export_tuple]["args"]
    try:
        return _read(io, _filetype_readers[extension], new_args, **kwargs)
    except ValueError:
        for extension_, in _hdx["export_types"]:
            try:
                return _read(io, _filetype_readers[extension_],
                    new_args, **kwargs)
            except ValueError as err:
                value_error = err
    error_text = ("The provided filepath, " + str(io)
        + ", is not compatable with the provided export_type, "
        + " ".join(export_tuple)
        + ". The export_type provided requires a filepath matching "
        + str(_hdx["files"][export_tuple]["filedialog"]["filetypes"])
        + "; the filepath appears to be a ." + extension + " file.")
    raise ValueError(error_text) from value_error


def hdx(filepath=None, export_type=None,
        drop_cols: str = "blank", **kwargs) -> pd.DataFrame:
    """Reads in a Quanterix HD-X results/history file.

    This function is a wrapper for pandas.read_excel and pandas.read_csv
    with additional options and defaults to make it straightforward to
    import a Sample Results Report or Run History file.

    Parameters
    ----------
    filepath : str, path object or file-like object, optional
        The path to the file to import. Any valid string path is
        acceptable. The string could be a URL. Valid URL schemes include
        http, ftp, s3, gs, and file. Can also be any os.PathLike or any
        object with a `read()` method. If not provided, a
        `tkinter.filedialog` opens, prompting the user to select a file.
    export_type : {"sample_results", "run_history"}, optional
        Type of Quanterix exported file. Options:

            - `"sample results"` : HD-X Sample Results Report

            - `"run history"` : HD-X Run History file

        If none is provided, the filetype will be inferred.
    drop_cols : {"blank", "uniform", "keep"}, default "blank"
        Should any columns be automatically dropped from the input file?
        Options:

            - `"blank"` : Drop all columns that are blank.

            - `"uniform"` : Drop all columns that have the same
              value for all rows, including blank columns.

            - `"keep"` : Do not drop any columns.

    **kwargs
        Additional arguments passed to pandas.read_excel or
        pandas.read_csv.

    Returns
    -------
    table : pandas.DataFrame
        Run History.

    See Also
    --------
    pandas.read_excel, pandas.read_csv : backend functions used by hdx

    """
    type_tuple = _hdx_export_type_tuple(export_type=export_type)
    io = _get_file(filepath, **_hdx["files"][type_tuple]["filedialog"])
    export_tuple = _hdx_ext_tuple(io) if type_tuple == ("any",) else type_tuple
    raw_table = _read_hdx_table(io, export_tuple, **kwargs)
    table = _cols_dropped(raw_table, drop_cols)
    return table


def plate_layout(io):
    return io
