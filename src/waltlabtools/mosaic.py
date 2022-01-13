"""Functions for analyzing MOSAIC data.

In addition to the dependencies for waltlabtools, waltlabtools.mosaic
also requires pandas 0.25 or greater and scikit-learn 0.21 or greater.

The public functions in waltlabtools.mosaic can be accessed via, e.g.,

.. code-block:: python

   import waltlabtools as wlt  # waltlabtools main functionality
   import waltlabtools.mosaic  # for analyzing MOSAIC assays

   subset_data = wlt.mosaic.plate_subsets()  # analyze data from a plate

if also using other functionality from the waltlabtools package, or

.. code-block:: python

   from waltlabtools import mosaic  # for analyzing MOSAIC assays

   subset_data = mosaic.plate_subsets()  # analyze data from a plate

if using only the waltlabtools.mosaic module.


-----


"""

import os
import re
import warnings
from tkinter import filedialog

from numpy.random import default_rng
import pandas as pd
import sklearn as sk
import sklearn.mixture

from .core import flatten, c4, _optional_dependencies, _DDOFS
from .cal_curve import CalCurve, limit_of_detection

if _optional_dependencies["jax"]:
    import jax.numpy as np
    from jax import jit
else:
    import numpy as np

    if _optional_dependencies["numba"]:
        from numba import jit
    else:
        from .core import jit


__all__ = [
    "PlateFileCollection",
    "mixture_orientation",
    "weighted_aeb",
    "log_transform",
    "mixture_aeb",
    "well_to_aeb",
    "extended_coefs",
    "plate_subsets",
]


# CONSTANTS
TOL = 1e-4
REG_COVAR = 1e-6
N_INIT = 5
MAX_ITER = 100
BACKGROUND_LEVEL = 2000
coefs_index = ["a", "b", "c", "d", "LOD", "LLOQ", "blank mean", "blank std", "blank cv"]


# CLASSES
class PlateFileCollection:
    """Collection of RCA product fluorescence intensity files.

    A PlateFileCollection object is a container for MOSAIC files. It is
    used to keep all wells from a given day, calibration curve, or assay
    together.

    Parameters
    ----------
    dir_path : str, optional
        The directory containing the MOSAIC plate files. If not provided
        a file dialog will be opened to select the folder.

    Attributes
    ----------
    name : str
        The name of the folder containing the MOSAIC plate files.
    wells : list
        The wells contained in the MOSAIC plate files, e.g., "A1".
    file_map : dict
        A dictionary mapping the well position to the file path for
        the corresponding flow cytometry output file.
    conc_map : dict
        A dictionary mapping the well position to the known
        concentration of the calibrator.
    dir_path
        The path to the folder containing the MOSAIC plate files.

    """

    def __init__(self, dir_path=None):
        if dir_path is None:
            dir_path_ = filedialog.askdirectory(
                title="Choose a Folder With Data for One Calibration Curve"
            )
        else:
            dir_path_ = dir_path

        layout_file = None
        well_files = []
        for file_entry in os.scandir(dir_path_):
            if file_entry.name.endswith(".xlsx") and ("layout" in file_entry.name):
                layout_file = file_entry.path
            elif file_entry.name.endswith(".csv"):
                well_files.append(file_entry)
        if layout_file is None:
            raise OSError("Could not find layout file in folder " + dir_path_ + ".")

        name = re.split("[/\\\\]", dir_path_)[-1].replace(" FINAL", "")

        file_map = {}
        conc_map = pd.read_excel(
            layout_file, header=None, index_col=0, squeeze=True
        ).to_dict()
        for file_entry in well_files:
            split_filename = re.split("[-–—_ .()\[\]|,*]", file_entry.name)
            for well in conc_map:
                if well in split_filename:
                    if file_entry.path not in file_map.values():
                        file_map[well] = file_entry.path
                    else:
                        file_map[file_entry.name] = file_entry.path

        if conc_map.keys() != file_map.keys():
            missing_files = set(conc_map.keys()) - set(file_map.keys())
            if missing_files:
                warnings.warn(
                    "In the folder '"
                    + name
                    + "', for the following wells, no CSV data files were found: '"
                    + "', '".join(missing_files)
                    + "'."
                )
                for key in missing_files:
                    del conc_map[key]

            missing_concs = set(file_map.keys()) - set(conc_map.keys())
            if missing_concs:
                warnings.warn(
                    "In the folder '"
                    + name
                    + "', the following wells were not found in the layout file: '"
                    + "', '".join(missing_concs)
                    + "'."
                )
                for key in missing_concs:
                    del file_map[key]

        self.name = name
        self.wells = sorted(list(conc_map.keys()))
        self.file_map = file_map
        self.conc_map = conc_map
        self.dir_path = dir_path_


@jit
def mixture_orientation(
    means_: np.ndarray, covariances_: np.ndarray, threshold_sds=5
) -> tuple:
    """Determines which peak is 'off.'

    Parameters
    ----------
    means_ : ndarray of length 2
        The two means of the Gaussian mixture model.
    covariances_ : ndarray of length 2
        The two covariances of the Gaussian mixture model.
    threshold_sds : float, default 5
        The number of standard deviations above the mean to use as a
        threshold for distinguishing on-beads in the case where there
        are very few on-beads.

    Returns
    -------
    off_label, sds_threshold, ordered_means
        Tuple of length 3. Its elements:
        - off_label : 1 or 2
            The label of the Gaussian mixture model with the lower mean.
        - sds_threshold : float
            The number of standard deviations above the mean to use as a
            threshold for distinguishing on-beads in the case where
            there are very few on-beads.
        - ordered_means : ndarray of length 2
            The two means of the Gaussian mixture model, ordered from
            lowest to highest (off to on).
    """
    off_label = np.argmin(means_)
    off_on = np.array([off_label, 1 - off_label])
    low_sd = np.sqrt(covariances_[off_label])
    sds_threshold = means_[off_label] + threshold_sds * low_sd
    ordered_means = means_[off_on]
    return off_label, sds_threshold, ordered_means


@jit
def weighted_aeb(onfrac_gmm, onfrac_sds):
    """Weighted average of the two measures of fraction on.

    Calculates the AEB using a weighted average of the Gaussian
    mixture model fraction on and the standard deviation-based fraction
    on.

    Parameters
    ----------
    onfrac_gmm : float
        The on-fraction of the Gaussian mixture model.
    onfrac_sds : float
        The on-fraction based on the standard deviation threshold.

    Returns
    -------
    aeb : float
        The average number of enzymes per bead.

    See Also
    --------
    aeb : calculate the AEB from a single f_on value

    """
    gmm_weight = (0.5 - 0.5 * np.cos(np.pi * onfrac_gmm)) ** 2
    onfrac = (1 - gmm_weight) * onfrac_sds + gmm_weight * onfrac_gmm
    return -np.log(1 - onfrac)


def log_transform(flat_data: np.ndarray) -> np.ndarray:
    """Log-transforms the data.

    Because a few values may be negative, a constant value is calculated
    to add to all of the values.

    Parameters
    ----------
    flat_data : 1D ndarray
        The data to be transformed.

    Returns
    -------
    log_data : 1D ndarray
        The log-transformed data.

    """
    first_percentile = np.quantile(flat_data, 0.01)
    const = first_percentile - BACKGROUND_LEVEL
    return np.log(flat_data - const)


def mixture_aeb(
    flat_data: np.ndarray,
    means_init=None,
    flat_len=None,
    threshold_sds=5,
    reg_covar=REG_COVAR,
) -> float:
    """Calculates AEB based on a 2-Gaussian mixture model.

    Parameters
    ----------
    flat_data : 1D ndarray
        Data used for fitting the mixture model. It is assumed that if
        the data should be log-transformed, they have already been
        transformed, e.g., with `log_transform`.
    means_init : array-like of length 2, optional
        The user-provided initial means, If not provided, means are
        initialized as the maximum and minimum of the data.
    flat_len : int, optional
        Length of flat_data. If not provided, it is calculated.
    threshold_sds : numeric, default 5
        The number of standard deviations above the mean to use as a
        threshold for distinguishing on-beads in the case where there
        are very few on-beads.
    reg_covar : numeric, default REG_COVAR
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    Returns
    -------
    float
        AEB, based on weighted average of two measures of calculating
        on-fraction.

    """
    mi = [np.amin(flat_data), np.amax(flat_data)] if means_init is None else means_init
    means_init_ = np.array(mi).reshape(-1, 1)
    mixture = sk.mixture.GaussianMixture(
        n_components=2,
        means_init=means_init_,
        tol=TOL,
        reg_covar=reg_covar,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        covariance_type="spherical",
    )
    try:
        gmm_labels = mixture.fit_predict(flat_data.reshape(-1, 1))
        off_label, sds_threshold, ordered_means = mixture_orientation(
            np.ravel(mixture.means_), np.ravel(mixture.covariances_), threshold_sds
        )
        flat_labels = np.ravel(1 - gmm_labels) if off_label else np.ravel(gmm_labels)
        corrected_flat_labels = (flat_labels & (flat_data > ordered_means[0])) | (
            flat_data > ordered_means[1]
        )
        flat_len_ = flat_len if flat_len is not None else len(flat_data)
        onfrac_gmm = np.sum(corrected_flat_labels) / flat_len_
        onfrac_sds = np.sum(flat_data > sds_threshold) / flat_len_
        return weighted_aeb(onfrac_gmm, onfrac_sds)
    except Exception as exc:
        if reg_covar < 100 * REG_COVAR:
            return mixture_aeb(
                flat_data=flat_data,
                flat_len=flat_len_,
                means_init=means_init,
                threshold_sds=threshold_sds,
                reg_covar=reg_covar * 2,
            )
        else:
            raise exc


def well_to_aeb(well_entry=None, log: bool = True, threshold_sds=5):
    """Calculates AEB of a well based on a 2-Gaussian mixture model.

    Parameters
    ----------
    well_entry : str, iterable, or os.DirEntry, optional
        The well to be analyzed. If not provided, a filedialog will
        open to select the output file for a well.
    log : bool, default True
        Should the data be log-transformed before fitting the Gaussian
        mixture model?
    threshold_sds : numeric, default 5
        The number of standard deviations above the mean to use as a
        threshold for distinguishing on-beads in the case where there
        are very few on-beads.

    Returns
    -------
    float or dict
        AEB, based on weighted average of two measures of calculating
        on-fraction. If multiple well_entry values are provided, a
        dictionary is returned with the AEB values for each well.
    """
    if isinstance(well_entry, str):
        data = pd.read_csv(well_entry)
    elif hasattr(well_entry, "__iter__"):
        return {
            entry: well_to_aeb(entry, log=log, threshold_sds=threshold_sds)
            for entry in well_entry
        }
    elif isinstance(well_entry, os.DirEntry):
        data = pd.read_csv(well_entry.path)
    else:
        well_path = filedialog.askopenfilenames(
            title="Choose a File with the Flow Cytometry Data for One Well",
            filetypes=[("Comma-Separated Values", "csv")],
        )
        return well_to_aeb(well_path)
    flat_data = log_transform(flatten(data)) if log else flatten(data)
    return mixture_aeb(flat_data=flat_data, threshold_sds=threshold_sds)


def extended_coefs(concs, aebs, corr="c4", cal_curve=None) -> dict:
    """Calculates the coefficients for a 4PL model.

    Parameters
    ----------
    concs : array-like
        The concentrations for the calibrators.
    aebs : array-like
        The AEBs for the calibrators.
    corr : {"n", "n-1", "n-1.5", "c4"} or numeric, default "c4"
        The sample standard deviation under-estimates the population
        standard deviation for a normally distributed variable.
        Specifies how this should be addressed. Options:

            - "n" : Divide by the number of samples to yield the
              uncorrected sample standard deviation.

            - "n-1" : Divide by the number of samples minus one to
              yield the square root of the unbiased sample variance.

            - "n-1.5" : Divide by the number of samples minus 1.5 to
              yield the approximate unbiased sample standard deviation.

            - "c4" : Divide by the correction factor to yield the
              exact unbiased sample standard deviation.

            - If numeric, gives the delta degrees of freedom.
    cal_curve : CalCurve, optional
        A CalCurve object to use. If not provided, a new CalCurve will
        be calculated based on the concs and aebs provided.

    Returns
    -------
    dict
        A dictionary of coefficients and properties of the calibration
        curve. Its elements are:

            "a", "b", "c", "d" : coefficients of the 4PL fit

            "LOD" : limit of detection

            "LLOQ" : lower limit of quantification (10 standard
            deviations above background)

            "blank mean" : mean AEB at 0 concentration

            "blank std" : standard deviation of AEB at 0 concentration

            "blank cv" : coefficient of variation of AEB at 0
            concentration

    """
    blank_array = flatten(aebs[concs == 0])
    if isinstance(cal_curve, CalCurve):
        new_cal_curve = cal_curve
        calibratible = True
    else:
        try:
            new_cal_curve = CalCurve.from_data(model="4PL", x=concs, y=aebs)
            calibratible = True
        except Exception:
            calibratible = False
    if calibratible:
        coefs = {key: value for key, value in new_cal_curve.coefs.items()}
        coefs["LOD"] = new_cal_curve.lod
        coefs["LLOQ"] = limit_of_detection(
            blank_array, new_cal_curve, sds=10, corr=corr
        )
    else:
        coefs = {
            "a": np.nan,
            "b": np.nan,
            "c": np.nan,
            "d": np.nan,
            "LOD": np.inf,
            "LLOQ": np.inf,
        }
    coefs["blank mean"] = np.mean(blank_array)
    try:
        ddof = _DDOFS[corr]
    except KeyError:
        ddof = float(corr)
    corr_factor = c4(len(blank_array)) if (corr == "c4") else 1
    coefs["blank std"] = np.std(blank_array, ddof=ddof) / corr_factor
    coefs["blank cv"] = coefs["blank std"] / coefs["blank mean"]
    return coefs


def plate_subsets(
    dir_path=None,
    save_aebs_to=None,
    save_coefs_to=None,
    log: bool = True,
    model="4PL",
    lod_sds=3,
    subsets: int = 10,
    sizes=(),
    corr="c4",
    threshold_sds=5,
) -> pd.DataFrame:
    """Calculates AEBs and coefficients for a plate.

    Parameters
    ----------
    dir_path : str, optional
        The directory containing the MOSAIC plate files. If not provided
        a file dialog will be opened to select the folder.
    save_aebs_to : str, optional
        The path to save the AEBs to.
    save_coefs_to : str, optional
        The path to save the coefficients to.
    log : bool, default True
        Should the data be log-transformed before fitting the Gaussian
        mixture model?
    model : str or Model, default "4PL"
        The model to fit.
    lod_sds : numeric, default 3
        The number of standard deviations above the mean to use as a
        limit of detection.
    subsets : int, default 10
        The number of subsets to create.
    sizes : iterable
        The number of beads in each subset. Technically optional, but
        the default argument is "()" which does not conduct any
        subsetting.
    corr : {"n", "n-1", "n-1.5", "c4"} or numeric, default "c4"
        The sample standard deviation under-estimates the population
        standard deviation for a normally distributed variable.
        Specifies how this should be addressed. Options:

            - "n" : Divide by the number of samples to yield the
              uncorrected sample standard deviation.

            - "n-1" : Divide by the number of samples minus one to
              yield the square root of the unbiased sample variance.

            - "n-1.5" : Divide by the number of samples minus 1.5 to
              yield the approximate unbiased sample standard deviation.

            - "c4" : Divide by the correction factor to yield the
              exact unbiased sample standard deviation.

            - If numeric, gives the delta degrees of freedom.
    Returns
    -------
    pd.DataFrame
        A dataframe of coefficients and extended parameters for each
        subset.

    See Also
    --------
    extended_coefs : for the coefficients and parameters returned

    """
    plate_files = PlateFileCollection(dir_path)
    concs = []
    datasets = []
    dlens = []
    aebs = []
    for well in plate_files.wells:
        concs.append(plate_files.conc_map[well])
        data = pd.read_csv(plate_files.file_map[well])
        flat_data = log_transform(flatten(data.values)) if log else flatten(data.values)
        dlens.append(len(flat_data))
        datasets.append(flat_data)
        aebs.append(mixture_aeb(flat_data=flat_data, threshold_sds=threshold_sds))
    concs_flat = flatten(concs)
    aebs_flat = flatten(aebs)
    cal_curve = CalCurve.from_data(
        model=model, x=concs_flat, y=aebs_flat, lod_sds=lod_sds, corr=corr
    )
    coefs = extended_coefs(concs_flat, aebs_flat, corr, cal_curve)
    coef_table = pd.DataFrame.from_dict(coefs, orient="index")
    aeb_table = pd.DataFrame.from_dict(
        {
            "Well": plate_files.wells,
            "Concentration": concs_flat,
            "AEB": aebs_flat,
            "Number of Beads": dlens,
        }
    )
    rng = default_rng()
    subset_aebs = {}
    for size in sizes:
        subset_aebs[size] = pd.DataFrame(
            columns=range(subsets), index=plate_files.wells, dtype=float
        )
        for w in range(len(concs)):
            if dlens[w] > size:
                for subset in range(subsets):
                    subset_data = rng.choice(datasets[w], size, replace=False)
                    subset_aebs[size].at[plate_files.wells[w], subset] = mixture_aeb(
                        flat_data=subset_data, threshold_sds=threshold_sds
                    )
            else:
                for subset in range(subsets):
                    subset_aebs[size].at[plate_files.wells[w], subset] = aebs_flat[w]
        subset_aebs[size]["mean"] = subset_aebs[size][range(subsets)].mean(
            axis=1, skipna=False
        )
        subset_aebs[size]["std"] = subset_aebs[size][range(subsets)].std(
            axis=1, skipna=False, ddof=1.5
        )
        subset_aebs[size]["median"] = subset_aebs[size][range(subsets)].median(
            axis=1, skipna=False
        )
        subset_aebs[size]["IQR 25%"] = subset_aebs[size][range(subsets)].quantile(
            q=0.25, axis=1
        )
        subset_aebs[size]["IQR 75%"] = subset_aebs[size][range(subsets)].quantile(
            q=0.75, axis=1
        )
    if save_aebs_to is None:
        save_aebs_to_ = filedialog.asksaveasfilename(
            initialfile=plate_files.name + " subset AEBs.xlsx",
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "xlsx")],
        )
    else:
        save_aebs_to_ = save_aebs_to
    if save_aebs_to_:
        with pd.ExcelWriter(save_aebs_to_) as writer:
            aeb_table.to_excel(writer, sheet_name="Original", index=False)
            for size in sizes:
                subset_aebs[size].to_excel(writer, sheet_name=str(size))
    subset_coefs = {}
    for size in sizes:
        subset_coefs[size] = pd.DataFrame(
            columns=range(subsets), index=coefs_index, dtype=float
        )
        for subset in range(subsets):
            one_subset_coefs = extended_coefs(
                concs_flat, subset_aebs[size][subset], corr
            )
            for key, value in one_subset_coefs.items():
                subset_coefs[size].at[key, subset] = value
        subset_coefs[size]["mean"] = (
            subset_coefs[size][range(subsets)].fillna(np.inf).mean(axis=1, skipna=False)
        )
        subset_coefs[size]["std"] = (
            subset_coefs[size][range(subsets)]
            .fillna(np.inf)
            .std(axis=1, skipna=False, ddof=1.5)
        )
        subset_coefs[size]["median"] = (
            subset_coefs[size][range(subsets)]
            .fillna(np.inf)
            .median(axis=1, skipna=False)
        )
        subset_coefs[size]["IQR 25%"] = (
            subset_coefs[size][range(subsets)].fillna(np.inf).quantile(q=0.25, axis=1)
        )
        subset_coefs[size]["IQR 75%"] = (
            subset_coefs[size][range(subsets)].fillna(np.inf).quantile(q=0.75, axis=1)
        )
    if save_coefs_to is None:
        save_coefs_to_ = filedialog.asksaveasfilename(
            initialfile=plate_files.name + " subset coefficients.xlsx",
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "xlsx")],
        )
    else:
        save_coefs_to_ = save_coefs_to
    if save_coefs_to_:
        with pd.ExcelWriter(save_coefs_to_) as writer:
            coef_table.to_excel(writer, sheet_name="Original", header=False)
            for size in sizes:
                subset_coefs[size].to_excel(writer, sheet_name=str(size))
    return coef_table
