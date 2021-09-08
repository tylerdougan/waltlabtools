# Functions for reading in instrument data.

import pandas as pd
from tkinter import filedialog


__all__ = ["run_history"]


# class FileType:
#     pass
# 
# 
# _run_history_aux = FileType()
# _run_history_aux.general_cols = {
#     "Sample Barcode",
#     "Assay",
#     "Plex",
#     "Location",
#     "Carrier Barcode",
#     "Unit",
#     "Estimated Time to Result",
#     "Completion Date",
#     "Batch Name",
#     "Sample Type",
#     "Dilution Factor",
#     "Dilution Description",
#     "Assay Revision",
#     "Batch ID",
#     "Calibration Curve ID",
#     "Instrument SN",
#     "Result ID",
#     "SW Version",
#     "Test Order ID"}
# 
# _run_history_aux.replicates_cols = {
#     "Replicate AEB",
#     "Replicate Conc.",
#     "Job Status",
#     "Job ID", 
#     "Flags",
#     "Errors",
#     "Fraction On",
#     "Isingle",
#     "Analysis Mode",
#     "Result Status",
#     "Image Quality Score",
#     "Ibead",
#     "Number of Beads",
#     "Analog AEB",
#     "Bead Concentration",
#     "Curve Name",
#     "Date Curve Created",
#     "Digital AEB",
#     "Extended Properties",
#     "Fraction Monomeric Beads",
#     "Job Start Cycle",
#     "Replicate Result ID",
#     "Used Reagents",
#     "User Name"}
# 
# _run_history_aux.statistics_cols = {
#     "Mean AEB",
#     "SD AEB",
#     "CV AEB",
#     "Mean Conc.",
#     "SD Conc.",
#     "CV Conc."}
# 
# _run_history_aux.details_cols = {
#     "Carrier Barcode",
#     "Estimated Time to Result",
#     "Completion Date",
#     "Job Status",
#     "Job ID",
#     "Assay Revision",
#     "Batch ID",
#     "Instrument SN",
#     "Job Start Cycle",
#     "Replicate Result ID",
#     "Result ID",
#     "SW Version",
#     "Test Order ID",
#     "Used Reagents",
#     "User Name"}


def _get_file(filepath, title: str, filetypes: list):
    """
    
    """
    if filepath is None:
        io = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    else:
        io = filepath
    return io


filetype_readers = {
    "csv": pd.read_csv,
    "excel": pd.read_excel,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "opendocument": pd.read_excel,
    "odf": pd.read_excel}
readers_set = set(filetype_readers.keys())


def _table_filetype(io, filetype=None) -> pd.DataFrame:
    if filetype is str:
        filetype_casefold = filetype.casefold()
        if filetype_casefold in filetype_readers.keys():
            return (filetype_readers[filetype_casefold](io),
                filetype_readers[filetype_casefold])
    for reader in readers_set:
        try:
            return reader(io), reader
        except Exception:
            pass
    raise UnicodeError("Pandas failed to read file " + str(io)
        + " with filetype " + str(filetype) + ".")


def _cols_dropped(raw_table: pd.DataFrame,
        drop_cols: {"blank", "uninformative", "keep"} = "blank"
        ) -> pd.DataFrame:
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


def run_history(filepath=None,
        drop_cols: {"blank", "uninformative", "keep"} = "blank"
        ) -> pd.DataFrame:
    """
    Reads in a Quanterix HD-X Run History file.
   
    Parameters
    ----------
    filepath : str, path object or file-like object, optional
        The path to the Run History CSV file. Any valid string path is
        acceptable. The string could be a URL. Valid URL schemes include
        http, ftp, s3, gs, and file. Can also be any os.PathLike or any
        object with a `read()` method. If not provided, a
        `tkinter.filedialog` opens, prompting the user to select a file.
    drop_cols : {"blank", "uniform", "keep"}, default "blank"
        Should any columns be dropped? Options:
            - `"blank"` : Drop all columns that are blank.
            - `"uninformative"` : Drop all columns that have the same
              value for all rows, which includes all blank columns.
            - `"keep"` : Do not drop any columns.

    Returns
    -------
    table : pandas.DataFrame
        Run History.
   
    """
    io = _get_file(filepath, title="Choose a Run History File",
        filetypes=[("Comma-Separated Values", "csv")])
    raw_table = pd.read_csv(io, header=0)
    table = _cols_dropped(raw_table, drop_cols)
    return table


# def sample_results(filepath=None,
#         drop_cols: {"blank", "uninformative", "keep"} = "blank"
#         ) -> pd.DataFrame:
#     """
#     Reads in a Quanterix HD-X Sample Results Report file.
#    
#     Parameters
#     ----------
#     filepath : str, path object or file-like object, optional
#         The path to the Run History CSV file. Any valid string path is
#         acceptable. The string could be a URL. Valid URL schemes include
#         http, ftp, s3, gs, and file. Can also be any os.PathLike or any
#         object with a `read()` method. If not provided, a
#         `tkinter.filedialog` opens, prompting the user to select a file.
#     drop_cols : {"blank", "uniform", "keep"}, default "blank"
#         Should any columns be dropped? Options:
#             - `"blank"` : Drop all columns that are blank.
#             - `"uninformative"` : Drop all columns that have the same
#               value for all rows, which includes all blank columns.
#             - `"keep"` : Do not drop any columns.
# 
#     Returns
#     -------
#     table : pandas.DataFrame
#         Sample Results Report.
#    
#     """
#     io = _get_file(filepath, title="Choose a Sample Results Report File",
#         filetypes=[("Excel 97–2004 Workbook", "xls")])
#     raw_table = pd.read_excel(io, header=0, skiprows=5)
#     table = _cols_dropped(raw_table, drop_cols)
#     return table
# 
# 
# def batch_calibration():
#     return None
# 
# 
# def plate_layout(filepath=None, filetype=None,
#         layout_cols: tuple = ("Plate", "Row", "Column"),
#         ) -> pd.DataFrame:
#     io = _get_file(filepath, title="Choose a Plate Layout File",
#         filetypes=[
#             ("Comma-Separated Values", "csv"),
#             ("Excel Workbook", "xlsx"),
#             ("Excel 97–2004 Workbook", "xls")])
#     raw_table, reader = _table_filetype(io, filetype)
#     if "Plate" in raw_table.iloc[0]:
        