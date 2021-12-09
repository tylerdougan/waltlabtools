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

import pandas as pd

from .core import _optional_dependencies

if _optional_dependencies["tkinter"]:
    from tkinter import filedialog
else:
    def filedialog(*args, **kwargs):
        """Return error if tkinter is not installed."""
        raise ModuleNotFoundError(
            "If tkinter is not installed, a filepath must be provided.")


__all__ = ["hdx"]


class _InstrumentType:
    pass


_hdx = _InstrumentType()

_hdx.files = {
    ("sample", "results"): {
        "title": "Choose a Sample Results Report File",
        "filetypes": [("Excel 97–2004 Workbook", "xls")]},
    ("run", "history"): {
        "title": "Choose a Run History File",
        "filetypes": [("Comma-Separated Values", "csv")]},
    ("any",): {
        "title": "Choose a Sample Results Report or Run History File",
        "filetypes": [
            ("Excel 97–2004 Workbook", "xls"),
            ("Comma-Separated Values", "csv"),
            ("All Files", "*")]},
}

_hdx.args = {
    ("sample", "results"): {
        "header": 5},
    ("run", "history"): {
        "header": 0},
    ("any",): {}
}

_hdx.export_types = {
    "xls": ("sample", "results"),
    "csv": ("run", "history")
}
_hdx.export_types["*"] = list(set(_hdx.export_types.values()))


def _get_file(filepath, **kwargs):
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
    if drop_cols == "keep":
        return raw_table
    elif drop_cols == "uninformative":
        uninformative_cols = []
        for colname in raw_table.columns:
            if len(raw_table[colname].unique()) <= 1:
                uninformative_cols.append(colname)
        return raw_table.drop(columns=uninformative_cols)
    else:
        return raw_table.dropna(axis="columns", how="all")


def _get_file_extension(io) -> str:
    if hasattr(io, "fspath"):
        return str(io.fspath()).rsplit('.', maxsplit=1)[-1]
    else:
        return str(io).rsplit('.', maxsplit=1)[-1]


def _hdx_export(export_type: str = "any") -> tuple:
    export_lowercase = str(export_type).casefold()
    for key in _hdx.files:
        if all([(word in export_lowercase) for word in key]):
            return key
    return ("any",)


def _reader(io, reader, new_args, **kwargs) -> pd.DataFrame:
    kwargs_ = {key: value for key, value in kwargs.items()
        if key in _reader_args[reader]}
    kwargs_.update(new_args)
    return reader(io, **kwargs_)


def _read_table(io, extension, export, **kwargs) -> pd.DataFrame:
    value_error = None
    if extension in _hdx.export_types:
        if export == _hdx.export_types[extension]:
            return _reader(io, _filetype_readers[extension],
                _hdx.args[export], **kwargs)
    elif (export == ("any",)) and (extension in _hdx.export_types):
        return _reader(io, _filetype_readers[extension],
            _hdx.args[_hdx.export_types[extension]], **kwargs)
    for extension_, export_ in _hdx.export_types.items():
        try:
            return _reader(io, _filetype_readers[extension_],
                _hdx.args[export_], **kwargs)
        except ValueError as err:
            value_error = err
            continue
    error_text = ("The provided filepath, " + str(io)
        + ", is not compatable with the provided export_type, "
        + " ".join(export)
        + ". The export_type provided requires a filepath matching "
        + str(_hdx.files[export]["filetypes"])
        + "; the filepath appears to be a ." + extension + " file.")
    raise ValueError(error_text) from value_error


def hdx(filepath=None, export_type=None,
        drop_cols: str = "blank", **kwargs) -> pd.DataFrame:
    """Reads in a Quanterix HD-X results/history file.

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

    """
    export = _hdx_export(export_type)
    io = _get_file(filepath, **_hdx.files[export])
    extension = _get_file_extension(io)
    if export == ("any",):
        if extension in _hdx.export_types:
            export = _hdx.export_types[extension]
        else:
            export = _hdx.export_types["*"]
    raw_table = _read_table(io, extension, export, **kwargs)
    table = _cols_dropped(raw_table, drop_cols)
    return table
