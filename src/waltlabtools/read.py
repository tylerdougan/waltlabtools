import os
import pathlib
from collections.abc import Callable, Hashable, Iterable
from io import IOBase
from mmap import mmap
from tkinter import Tk, filedialog
from typing import Any, Optional

import pandas as pd

from .cal_curve import CalCurve, _CalCurveSeries
from .core import match_kwargs
from .model import MODELS, Model


def _read_tsv(*args, **kwargs):
    kwargs.setdefault("delimiter", "\t")
    return pd.read_csv(*args, **kwargs)


_EXTENSION_READERS: dict[str, Callable] = {
    ".csv": pd.read_csv,
    ".tsv": _read_tsv,
    ".xls": pd.read_excel,
    ".xlsb": pd.read_excel,
    ".xlsm": pd.read_excel,
    ".xlsx": pd.read_excel,
    "": pd.read_csv,
}

_SAMPLE_RESULTS_COLNAMES: dict[str, str] = {
    "AEB": "Replicate AEB",
    "Concentration": "Replicate Conc.",
    "Mean Concentration": "Mean Conc.",
    "SD Concentration": "SD Conc.",
    "CV Concentration": "CV Conc.",
}

_HDX_EXCEPTIONS: dict[str, Any] = {
    "TooMuchFluorescenceInResorufinChannelException": 30,
}


def _suffix(filepath_or_buffer: os.PathLike | str | IOBase) -> str:
    """Determines the file extension for a given filepath or buffer.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        The filepath or buffer to determine the file extension for.

    Returns
    -------
    str
        The file extension, including the leading dot (e.g., '.txt').

    Raises
    ------
    TypeError
        If the input is not a valid filepath or buffer.
    """
    if hasattr(filepath_or_buffer, "suffix"):
        return filepath_or_buffer.suffix  # type: ignore
    elif isinstance(filepath_or_buffer, str):
        i = filepath_or_buffer.rfind(".")
        return filepath_or_buffer[i:] if (0 < i < len(filepath_or_buffer) - 1) else ""
    elif isinstance(filepath_or_buffer, IOBase):
        return ""
    else:
        raise TypeError(
            "Could not determine file extension for file "
            f"{filepath_or_buffer} of type {type(filepath_or_buffer)}."
        )


def _select_files_or_folders(folder: bool = False, multiple: bool = True) -> list[str]:
    """Displays a dialog box to select files or a folder.

    Parameters
    ----------
    folder : bool, default False
        If False (default), the dialog box will allow selecting files.
        If True, the dialog box will allow selecting a folder.
    multiple : bool, default True
        If True (default), the dialog box will allow selecting multiple
        files. If False, the dialog box will only allow selecting a
        single file. Ignored if folder is True.

    Returns
    -------
    io
        A list of strings representing the paths of the selected files
        or folder.

    Notes
    -----
    This function uses the `Tk` and `filedialog` modules from the
    `tkinter` package to display the dialog box. A hidden `Tk` window is
    created to facilitate the selection, and it is destroyed after the
    selection is made. The dialog box may be hidden behind other
    windows.

    """
    root = Tk()
    root.withdraw()

    if folder:
        io = [filedialog.askdirectory()]
    elif multiple:
        io = list(filedialog.askopenfilenames())
    else:
        io = [filedialog.askopenfilename()]

    root.destroy()
    return io


def _accumulate_list(iterable: Iterable, func: Callable) -> list:
    """Applies a function to iterable items and accumulates a list.

    Applies a given function to each item in an iterable and
    accumulates the results in a single list. The function should
    return an iterable for each item in the input iterable. The
    function is applied to each item, and the returned iterables are
    combined into a single list.

    Parameters
    ----------
    iterable : Iterable
        An iterable containing items that the function `func` will be
        applied to.
    func : Callable
        A function that takes an item from the iterable as input and
        returns an iterable.

    Returns
    -------
    list
        A list containing the combined results of applying the function
        `func` to each item in the input iterable.
    """
    result = []
    for item in iterable:
        result.extend(func(item))
    return result


def _is_leaf(io) -> bool:
    """Check if an object is a single file-like object."""
    return isinstance(io, (str, bytes, mmap, os.PathLike)) or not isinstance(
        io, Iterable
    )


def crawl(io) -> list:
    """Traverses directories and iterables to assemble a list of files.

    If a filepath (e.g., a string or an os.PathLike) or buffer is
    passed, it will be returned as a list. If a directory path is
    passed, then a list of all of its constituent files will be
    returned. Finally, if a collection of filepaths, buffers, or
    dictionaries is passed, then a list of all of their constituent
    files will be returned.

    Parameters
    ----------
    io : Any
        Any valid string path is acceptable. The string could be a URL.
        Valid URL schemes include http, ftp, s3, gs, and file. For file
        URLs, a host is expected. A local file could be:
        file://localhost/path/to/table.csv. If you want to pass in a
        path object, crawl accepts any os.PathLike. By file-like
        object, we refer to objects with a read() method, such as a
        file handle (e.g. via builtin open function) or StringIO. If
        the path of a directory is passed, it will be traversed and a
        list of its constituent files will be returned. If a folder or
        collection of files is passed, then each element of the
        collection will be crawled.

    Returns
    -------
    list
        A list of files or buffers. If `io` is a directory, then this
        list contains all files inside of it and its subfolders. If
        `io` is a collection of files, buffers, and directories, then
        this list contains all files and buffers in the collection, and
        all files inside of directories in the collection.
    """

    if hasattr(io, "read") or hasattr(io, "write"):
        return [io]

    elif _is_leaf(io):
        try:
            io_path = pathlib.Path(io)
        except TypeError:
            return [io]
        if io_path.is_dir():
            return _accumulate_list(io_path.iterdir(), crawl)
        return [io_path]

    else:
        return _accumulate_list(io, crawl)


def _get_sample_barcode(row: pd.Series) -> Hashable:
    """Gets the sample ID from a given row of a pandas DataFrame.

    If the "Sample Barcode" field is missing, the sample ID is
    constructed using a tuple of the "Batch Name" and "Location"
    fields. Otherwise, the "Sample Barcode" value is used as the
    sample ID.

    Parameters
    ----------
    row : pd.Series
        A row from self.raw.

    Returns
    -------
    str or tuple of str
        The sample ID.
    """
    if (
        ("Sample Barcode" in row)
        and pd.notna("Sample Barcode")
        and (row["Sample Barcode"] != "")
    ):
        return row["Sample Barcode"]
    else:
        return (row["Batch Name"], row["Location"])


class HDX:
    """Quanterix HD-X file reader.

    Reads in data from one or more Quanterix HD-X run histories
    (preferred) or sample results reports. The data are combined into a
    pandas.DataFrame called `raw`, and the assays are identified. From
    there, calibration curves can be fit to the data, and the data can
    be tidied.

    Parameters
    ----------
    io : Any, optional
        Any valid string path is acceptable. The string could be a URL.
        Valid URL schemes include http, ftp, s3, gs, and file. For file
        URLs, a host is expected. A local file could be:
        file://localhost/path/to/table.csv. If you want to pass in a
        path object, crawl accepts any os.PathLike. By file-like
        object, we refer to objects with a read() method, such as a
        file handle (e.g. via builtin open function) or StringIO. If
        the path of a directory is passed, it will be traversed and a
        list of its constituent files will be returned. If a list,
        tuple, set, dictionary, or generator is passed, then each
        element of the collection will be crawled. If not provided, a
        dialogue box will open, asking the user to select files.

    Examples
    --------
    To read in and combine Quanterix run histories and sample results
    reports chosen from a dialogue box, call without any arguments:

    >>> q = wlt.HDX()

    """

    def __init__(
        self,
        io=None,
        raw: Optional[pd.DataFrame | Iterable[pd.DataFrame | pd.Series]] = None,
        cal_curves: Optional[pd.Series] = None,
        **kwargs,
    ):
        if raw is None:
            # initialize attributes
            file_extensions = kwargs.get("file_extensions", [".csv", ".xls"])

            # read in data files
            if io is None:
                io = _select_files_or_folders(
                    folder=kwargs.get("folder", False),
                    multiple=kwargs.get("multiple", True),
                )
            filepaths = crawl(io)

            # determine which kwargs from __init__ to pass to each reader
            readers = {
                key: value
                for key, value in _EXTENSION_READERS.items()
                if key in file_extensions
            }
            reader_kwargs = {
                suffix: match_kwargs(reader, kwargs)
                for suffix, reader in readers.items()
            }

            # read in a DataFrame for each file
            raw_dfs = []
            for filepath in filepaths:
                suffix = _suffix(filepath)
                if suffix in readers:
                    raw_dfs.append(
                        readers[suffix](filepath, **reader_kwargs[suffix]).rename(
                            columns=_SAMPLE_RESULTS_COLNAMES
                        )
                    )

            # concatenate the DataFrames
            self.raw = pd.concat(raw_dfs, ignore_index=True).drop_duplicates()
        elif isinstance(raw, pd.DataFrame):
            self.raw = raw
        elif isinstance(raw, Iterable):
            self.raw = pd.concat(raw, ignore_index=True).drop_duplicates()
        else:
            raise ValueError(
                "Parameter `raw` must be a DataFrame or an iterable of DataFrames if provided."
            )
        self.raw["Sample Barcode"] = self.raw.apply(_get_sample_barcode, axis=1)  # type: ignore

        self._cal_curves = cal_curves
        self._tidy = None

        # Find and enumerate assays
        self.data_cols = kwargs.get("data_cols", ["Replicate AEB"])

        self.assay_defining_cols = kwargs.get("assay_defining_cols", ["Assay", "Plex"])

        if "assays" in kwargs:
            self.assays = kwargs["assays"]
        else:
            assays = self.raw.value_counts(subset=self.assay_defining_cols).index
            levels = reversed(assays.names[1:])
            for name in levels:
                dropped = assays.droplevel(name)
                if (assays.get_level_values(name).value_counts().shape[0] <= 1) or (
                    (name == assays.names[-1]) and dropped.nunique() == assays.nunique()
                ):
                    assays = dropped
            self.assays = assays.unique()

    # Calibration curve methods: calculate_cal_curves, cal_curves

    def calculate_cal_curves(
        self,
        model: str | Model | dict | pd.Series = "4PL",
        X_name: str = "Replicate Conc.",
        y_name: str = "Replicate AEB",
        force: bool = False,
        include_assays: Optional[Iterable] = None,
        exclude_assays: Optional[Iterable] = None,
        **kwargs,
    ) -> pd.Series:
        # get the list of assays to generate curves for
        if isinstance(model, Model) or model in MODELS:
            if include_assays is not None:
                if isinstance(include_assays, (str, tuple)):
                    assay_models = {include_assays: model}
                else:
                    assay_models = {assay: model for assay in include_assays}
            elif exclude_assays is not None:
                if isinstance(exclude_assays, (str, tuple)):
                    assay_models = {
                        assay: model for assay in self.assays if assay != exclude_assays
                    }
                else:
                    assay_models = {
                        assay: model
                        for assay in self.assays
                        if assay not in exclude_assays
                    }
            else:
                assay_models = {assay: model for assay in self.assays}
        elif hasattr(model, "items"):
            assay_models = model
        else:
            raise ValueError("Model not found.")

        # generate the curves
        cc_init_kwargs = match_kwargs(CalCurve, kwargs)
        cc_fit_kwargs = match_kwargs(CalCurve.fit, kwargs)
        self._cal_curves = _CalCurveSeries(index=self.assays, dtype=object)

        for assay, model in assay_models.items():  # type: ignore
            if isinstance(assay, tuple):  # MultiIndex
                indexer = pd.DataFrame(
                    {
                        name: self.raw[name] == level_value
                        for name, level_value in zip(self.assays.names, assay)
                    }
                ).all(axis=1)
            else:  # 1D Index
                indexer = self.raw[self.assays.name] == assay
            indexer = indexer & (self.raw["Sample Type"] == "Calibrator")

            X = self.raw.loc[indexer, X_name]
            y = self.raw.loc[indexer, y_name]
            try:
                self._cal_curves[assay] = CalCurve(model=model, **cc_init_kwargs).fit(
                    X=X, y=y, **cc_fit_kwargs
                )
            except Exception as e:
                if force:
                    raise RuntimeError(f"Error on assay {assay}.") from e
        return self._cal_curves

    @property
    def cal_curves(self) -> pd.Series:
        if self._cal_curves is None:
            self._cal_curves = self.calculate_cal_curves()  # Use default parameters
        return self._cal_curves

    # Make concentration

    def _calculate_concentrations(
        self,
        colname: Hashable = "Replicate AEB",
        newname: Optional[str] = None,
        fix_hdx_exceptions: bool = False,
        **kwargs,
    ) -> pd.Series:
        "Create a new column in self.raw and calculate concentrations."
        if (self._cal_curves is None) or kwargs:
            self.calculate_cal_curves(**kwargs)

        if fix_hdx_exceptions:

            def replicate_aeb(row):  # type: ignore
                if pd.isna(row[colname]) and row["Errors"]:
                    for key in _HDX_EXCEPTIONS:
                        if key in row["Errors"]:
                            return _HDX_EXCEPTIONS[key]
                return row[colname]

        else:

            def replicate_aeb(row):
                return row[colname]

        if len(self.assays.names) > 1:

            def get_assay(row):  # type: ignore
                return tuple(row[self.assays.names])
        else:

            def get_assay(row):
                return row[self.assays.name]

        def apply_cal_curve(row: pd.Series) -> float:
            assay = get_assay(row)
            return self.cal_curves[assay].estimate(replicate_aeb(row))

        if newname is None:
            newname = f"Concentration Calculated from {colname}"
        self.raw[newname] = self.raw.apply(apply_cal_curve, axis=1)
        return self.raw[newname]

    # Tidy data methods: _make_tidy, tidy

    def calculate_tidy(
        self,
        stat: str | Callable = "median",
        colname: Optional[str] = None,
        use_curves: bool = False,
        **kwargs,
    ):
        if colname is None:
            colname = "Replicate AEB" if use_curves else "Replicate Conc."
        specimens = self.raw
        if use_curves:
            colname = self._calculate_concentrations(colname=colname, **kwargs).name  # type: ignore

        tidy_columns = ["Sample Barcode", *list(self.assays.names), colname]
        specimens = self.raw[self.raw["Sample Type"] == "Specimen"][tidy_columns]

        self._tidy = (
            specimens.groupby(["Sample Barcode", *self.assays.names])
            .agg(stat)
            .reset_index()
            .pivot_table(
                index="Sample Barcode", columns=self.assays.names, values=colname
            )
        )

        return self._tidy

    @property
    def tidy(self) -> pd.DataFrame:
        if self._tidy is None:
            self.calculate_tidy()
        return self._tidy  # type: ignore

    def __add__(self, other):
        if not isinstance(other, HDX):
            raise TypeError("Unsupported operand type for +")

        return HDX(raw=[self.raw, other.raw])

    def __eq__(self, other) -> bool:
        """Two HDX objects are equal if they have the same raw values."""
        if not isinstance(other, HDX):
            raise TypeError("Unsupported operand type for ==")

        if len(self.raw) != len(other.raw):
            return False

        for column in self.raw.columns:
            if column in other.raw.columns and not self.raw[column].equals(
                other.raw[column]
            ):
                return False

        return True
