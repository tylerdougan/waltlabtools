# Functions for reading in instrument data.

import pandas as pd


class FileType:
    pass


_run_history_aux = FileType()
_run_history_aux.general_cols = {
    "Sample Barcode",
    "Assay",
    "Plex",
    "Location",
    "Carrier Barcode",
    "Unit",
    "Estimated Time to Result",
    "Completion Date",
    "Batch Name",
    "Sample Type",
    "Dilution Factor",
    "Dilution Description",
    "Assay Revision",
    "Batch ID",
    "Calibration Curve ID",
    "Instrument SN",
    "Result ID",
    "SW Version",
    "Test Order ID"}

_run_history_aux.replicates_cols = {
    "Replicate AEB",
    "Replicate Conc.",
    "Job Status",
    "Job ID", 
    "Flags",
    "Errors",
    "Fraction On",
    "Isingle",
    "Analysis Mode",
    "Result Status",
    "Image Quality Score",
    "Ibead",
    "Number of Beads",
    "Analog AEB",
    "Bead Concentration",
    "Curve Name",
    "Date Curve Created",
    "Digital AEB",
    "Extended Properties",
    "Fraction Monomeric Beads",
    "Job Start Cycle",
    "Replicate Result ID",
    "Used Reagents",
    "User Name"}

_run_history_aux.statistics_cols = {
    "Mean AEB",
    "SD AEB",
    "CV AEB",
    "Mean Conc.",
    "SD Conc.",
    "CV Conc."}

_run_history_aux.details_cols = {
    "Carrier Barcode",
    "Estimated Time to Result",
    "Completion Date",
    "Job Status",
    "Job ID",
    "Assay Revision",
    "Batch ID",
    "Instrument SN",
    "Job Start Cycle",
    "Replicate Result ID",
    "Result ID",
    "SW Version",
    "Test Order ID",
    "Used Reagents",
    "User Name"}


def hdx_run_history(filepath,
        include_replicates: bool = True,
        include_statistics: bool = True,
        drop_cols: {"blank", "uninformative", "keep"} = "blank"
        ) -> pd.DataFrame:
    """
    Read in a Quanterix HD-X Run History file.
   
    Parameters
    ----------
    `filepath` : str, path object or file-like object
        The path to the Run History CSV file. Any valid string path is
        acceptable. The string could be a URL. Valid URL schemes include
        http, ftp, s3, gs, and file. Can also be any os.PathLike or any
        object with a `read()` method.
    `replicates` : bool, default True
        Whether to include the results from individual replicates.
    `statistics` : bool, default True
        Whether to include summary statistics from each sample/plex,
        such as means, standard deviations, and coefficients of
        variation.
    `drop_cols` : {"blank", "uniform", "keep"}, default "blank"
        Should uninformative columns be dropped? Options:
            - `"blank"` : Drop all columns that are blank.
            - `"uninformative"` : Drop all columns that only have a
            single value for all rows, which includes all blank columns.
            - `"keep"` : Do not drop any columns.

    Returns
    -------
    `` : 1D ndarray, list, or primitive
        Flattened version of `data`. If `on_bad_data="error"`, always
        an ndarray.
   
    """

    skip_blank_lines = not drop_cols == "keep"
    table = pd.read_csv(filepath, skip_blank_lines=skip_blank_lines)
    if not replicates:
        table.dropna(how="all",
            subset=_run_history_aux.statistics_cols, inplace=True)
    if not statistics:
        table.dropna(how="all",
            subset=_run_history_aux.replicates_cols, inplace=True)
    if drop_cols == "blank":
        table.dropna(axis="columns", how="all", inplace=True)
    elif drop_cols == "uninformative":
        uninformative_cols = []
        for colname in table.columns:
            if len(table[colname].unique()) <= 1:
                uninformative_cols.append(colname)
        table.drop(columns=uninformative_cols, inplace=True)
    return table


def simoa_sample_results_report():
    return None


def simoa_batch_calibration_report():
    return None


