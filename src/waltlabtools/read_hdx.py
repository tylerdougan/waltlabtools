import os
from tkinter import filedialog
from collections import OrderedDict

from .core import _optional_dependencies

if _optional_dependencies["jax"]:
    import jax.numpy as np
else:
    import numpy as np

import pandas as pd

from .cal_curve import CalCurve
from .model import Model, models


def _read_tsv(*args, **kwargs):
    """Reads a tsv file. See `pandas.read_csv` for more information."""
    kwargs["sep"] = "\t"
    return pd.read_csv(*args, **kwargs)


_PIVOT_KWARGS = 


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
        io = filedialog.askopenfilenames(**kwargs)
    else:
        io = filepath
    return io


def _read_pd(filepath, **kwargs):
    """Reads a spreadsheet file into a pandas dataframe.

    Automatically chooses the correct pandas read function based on
    file extension, and then returns the dataframe generated from the
    data.

    Parameters
    ----------
    filepath : str, path object or file-like object
        The path to the file to import. Any valid string path is
        acceptable. The string could be a URL. Valid URL schemes include
        http, ftp, s3, gs, and file. Can also be any os.PathLike or any
        object with a `read()` method.

    Returns
    -------
    pandas.DataFrame
    """
    file_extension = os.fspath(filepath).split(".")[-1]
    return _PD_READERS[file_extension](filepath, **kwargs), file_extension


def read_hdx_file(
    filepath=None,
    cal_curve=None,
    **kwargs
):
    """[summary]

    [extended_summary]

    Parameters
    ----------
    filepath : [type], optional
        [description], by default None
    cal_curve : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]

    Other Parameters
    ----------------

    """
    pivot_table_kwargs = {key: value for key, value in kwargs.items() if key in }

    io = _get_file(filepath, title="Choose a Sample Results Report or Run History File",
        filetypes=[("Sample Results Report", "xls"), ("Run History", "csv"), ("All Files", "*")],)
    raw_df, file_extension = _read_pd(io, **kwargs)

    if cal_curve is None:
        raw_df.reindex(columns=lambda colname: "x" if colname == "Replicate Conc." else colname, inplace=True)
    if isinstance(cal_curve, CalCurve):
        raw_df["x"] = raw_df["Replicate AEB"].applymap(cal_curve.inverse)
    elif isinstance(cal_curve, Model) or (cal_curve in models):
        cal_curve_ = CalCurve.from_data(
            x=raw_df["Replicate Conc."][raw_df["Sample Type"] == "Calibrator"],
            y=raw_df["Replicate AEB"][raw_df["Sample Type"] == "Calibrator"],
            model=cal_curve,
            **kwargs #TODO: refine kwargs here
        )
        raw_df["x"] = raw_df["Replicate AEB"].applymap(cal_curve_.inverse)

    validated_kwargs = {
        "values": "x",
        "index": "Sample Barcode",
        "columns": "Plex",
        "sort": False
    }
    validated_kwargs.update(kwargs) #TODO: refine kwargs here
    return raw_df.pivot_table(**validated_kwargs)

    

################################################################


################################################################


################################################################


################################################################


class DataFrame(pd.DataFrame):
    """Pandas DataFrame augmented to be compatable with waltlabtools.

    The waltlabtools.DataFrame class is designed for sample results
    reports and run histories. The general idea is that it is a normal
    pandas.DataFrame for all duck typing purposes, but it carries a
    few pieces of additional information that will make it easier to
    transform into a data matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        The xls/csv file, imported as a pandas dataframe.
    x : column name, dict, or OrderedDict
        The readout data. If one column name is passed, it will be used
        as the data column. If a dictionary is passed, its keys should
        be column names and its values should be transformation
        functions. The first member of the dictionary which is not nan
        is used. To guarantee order, use an OrderedDict (from the
        collections model in the Python standard library) instead of a
        dict. For example, the following are equivalent:

            - x = "AEB"

            - x = {"AEB": waltlabtools.Id}

            - x = {"AEB": lambda x: x}

            - x = OrderedDict("AEB": waltlabtools.Id)

            - x = OrderedDict("AEB": lambda x: x)

    biomarker: column name or collection, default "Plex"
        The column(s) identifying the biomarker being measured for a
        given job. Defaults to "Plex", which uniquely specifies each
        assay/plex in most sample results reports and run histories.
    sample_type: column name, default "Sample Type"
        The column identifying the type of sample being measured
        (typically "Calibrator" or "Specimen"). Defaults to
        "Sample Type", the column name in sample results reports and run
        histories.

    """

    def __init__(self, df, x, biomarker="Plex", sample_type="Sample Type"):
        super().__init__(df)
        self.x = x
        self.biomarker = biomarker
        self.sample_type = sample_type


def sample_results(filepath=None, **kwargs) -> DataFrame:
    """Reads in a Quanterix HD-X sample results report xls file.

    Parameters
    ----------
    filepath : str, path object or file-like object, optional
        The path to the sample results report xls file to import.
        Any valid string path is acceptable. The string could be a URL.
        Valid URL schemes include http, ftp, s3, gs, and file. Can also
        be any os.PathLike or any object with a `read()` method. If not
        provided, a `tkinter.filedialog` opens, prompting the user to
        select a file.

    Returns
    -------
    DataFrame
        [description]
    """
    io = _get_file(
        filepath,
        title="Choose a Sample Results Report File",
        filetypes=[("Excel 97–2004 Workbook", "xls"), ("All Files", "*")],
    )
    x = kwargs.pop("x", "Concentration")
    biomarker = kwargs.pop("biomarker", "Plex")
    df = pd.read_excel(io, header=5, **kwargs)
    return DataFrame(df=df, x=x, biomarker=biomarker)


def run_history(filepath=None, **kwargs) -> DataFrame:
    """Reads in a Quanterix HD-X run history csv file.

    Parameters
    ----------
    filepath : str, path object or file-like object, optional
        The path to the run history csv file to import.
        Any valid string path is acceptable. The string could be a URL.
        Valid URL schemes include http, ftp, s3, gs, and file. Can also
        be any os.PathLike or any object with a `read()` method. If not
        provided, a `tkinter.filedialog` opens, prompting the user to
        select a file.

    Returns
    -------
    DataFrame
        [description]
    """
    io = _get_file(
        filepath,
        title="Choose a Run History File",
        filetypes=[("Comma-Separated Values", "csv"), ("All Files", "*")],
    )
    x = kwargs.pop("x", "Concentration")
    biomarker = kwargs.pop("biomarker", "Plex")
    df = pd.read_csv(io, header=0, **kwargs)
    return DataFrame(df=df, x=x, biomarker=biomarker)


# def _find_col(columns, colname=None, search_terms=()):
#     """Finds a matching column name.
#
#     If ``colname`` is provided and
#
#     Parameters
#     ----------
#     columns : [type]
#         [description]
#     colname : [type], optional
#         [description], by default None
#     lowercase_search_terms : tuple, optional
#         [description], by default ()
#
#     Returns
#     -------
#     [type]
#         [description]
#
#     Raises
#     ------
#     ValueError
#         [description]
#     """
#     all_columns = columns.colums if hasattr(columns, "columns") else columns
#
#     if (colname is not None) and (colname in all_columns):
#         return colname
#
#     lowercase_columns = [str(column).casefold() for column in all_columns]
#
#     if isinstance(search_terms, str):
#         lowercase_search_terms = search_terms.casefold()
#     for word in lowercase_search_terms:
#         possible_cols = [
#             column for column in lowercase_columns if word in column
#         ]
#         if len(possible_cols) == 1:
#             return possible_cols[0]
#
#     raise ValueError(
#         "Column not found. Expected a column containing "
#         + ", ".join(lowercase_search_terms)
#         + ", but there was no single unambiguous match."
#     )


def plate_layout(
    filepath=None,
    barcode_col=None,
    plate_col=None,
    row_col=None,
    column_col=None,
    well_col=None,
    **kwargs
) -> dict:
    """Reads a plate layout file mapping wells to sample barcodes.

    The plate layout file may be any spreadsheet type (e.g., csv, xls,
    xlsx). There should be one column for sample barcodes, and columns
    that uniquely specify each well, e.g., a "Plate" column and a "Well"
    column, or a "Plate" column, a "Row" column, and a "Column"
    (of the plate) column.

    Parameters
    ----------
    filepath : [type], optional
        [description], by default None
    barcode_col : [type], optional
        [description], by default None
    plate_col : [type], optional
        [description], by default None
    row_col : [type], optional
        [description], by default None
    column_col : [type], optional
        [description], by default None
    well_col : [type], optional
        [description], by default None

    Returns
    -------
    dict
        [description]
    """
    io = _get_file(filepath)
    layout_df = _read_pd(io, **kwargs)

    final_barcode_col = _find_col(
        layout_df.columns,
        colname=barcode_col,
        lowercase_search_terms=("barcode", "sample"),
    )

    # TODO: define layout
    return {}


def extract_data(self: DataFrame, **kwargs) -> pd.DataFrame:
    """Generates a dataframe of biomarker readings for each patient.

    Parameters
    ----------
    self : DataFrame
        [description]

    Returns
    -------
    extracted_df : pandas.DataFrame
        A new pandas.DataFrame whose columns are different biomarkers
        and whose rows are different samples.

    """
    validated_kwargs = {
        "values": self.x,
        "index": self.barcode,
        "columns": self.biomarker,
        "sort": False,
    }
    validated_kwargs.update(kwargs)
    return self.pivot_table(**validated_kwargs)
