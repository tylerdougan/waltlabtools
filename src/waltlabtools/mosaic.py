# IMPORTS
import os
import warnings
import re

import jax.numpy as np
from jax import jit
from numpy.random import default_rng
import pandas as pd
import sklearn as sk
import sklearn.mixture
from tkinter import filedialog

from .core import flatten, CalCurve, lod, c4


__all__ = ["PlateFileCollection", "mixture_orientation", "aeb",
    "log_transform", "gaussians", "mixture_aeb", "well_to_aeb",
    "extended_coefs", "plate_subsets"]


# CONSTANTS
tol = 1e-4
REG_COVAR = 1e-6
n_init = 5
max_iter = 100
coefs_index = ["a", "b", "c", "d", "LOD", "LLOQ",
    "blank mean", "blank std", "blank cv"]


# CLASSES
class PlateFileCollection:
    def __init__(self, dir_path=None):
        if dir_path is None:
            dir_path_ = filedialog.askdirectory(
                title="Choose a Folder With Data for One Calibration Curve")
        else:
            dir_path_ = dir_path
        unix_folder_name = dir_path_.split("/")[-1]
        windows_folder_name = dir_path_.split("\\")[-1]
        if len(windows_folder_name) < len(unix_folder_name):
            name = windows_folder_name.replace(" FINAL", "")
        else:
            name = unix_folder_name.replace(" FINAL", "")
        file_map = {}
        layout_file = None
        well_files = []
        for file_entry in os.scandir(dir_path_):
            if (file_entry.name.endswith(".xlsx") 
                    and ("layout" in file_entry.name)):
                layout_file = file_entry.path
            elif file_entry.name.endswith(".csv"):
                well_files.append(file_entry)
        if layout_file is None:
            raise OSError("Could not find layout file in folder " + dir_path_)
        else:
            conc_map = pd.read_excel(layout_file, header=None, index_col=0,
                                 squeeze=True).to_dict()
            conc_keys = set(conc_map.keys())
        for file_entry in well_files:
            split_filename = re.split("[-–—_ .()\[\]|,*]", file_entry.name)
            for well in conc_keys:
                if well in split_filename:
                    if file_entry.path not in file_map.values():
                        file_map[well] = file_entry.path
                    else:
                        file_map[file_entry.name] = file_entry.path
        if conc_map.keys() != file_map.keys():
            missing_files = set(conc_map.keys()) - set(file_map.keys())
            missing_concs = set(file_map.keys()) - set(conc_map.keys())
            print(name)
            if missing_files:
                print("CSV files missing:", *missing_files)
                for key in missing_files:
                    del conc_map[key]
            if missing_concs:
                print("Wells missing in layout spreadsheet:", *missing_concs)
                for key in missing_concs:
                    del file_map[key]
            print("")
        self.name = name
        self.wells = sorted(list(conc_map.keys()))
        self.file_map = file_map
        self.conc_map = conc_map
        self.dir_path = dir_path_


@jit
def mixture_orientation(means_, covariances_, threshold_sds=5):
    """Determine which peak is 'off.'
    """
    off_label = np.argmin(means_)
    off_on = np.array([off_label, 1-off_label])
    low_sd = np.sqrt(covariances_[off_label])
    sds_threshold = means_[off_label] + threshold_sds*low_sd
    ordered_means = means_[off_on]
    return off_label, sds_threshold, ordered_means


@jit
def aeb(onfrac_gmm, onfrac_sds):
    gmm_weight = (0.5 - 0.5*np.cos(np.pi * onfrac_gmm))**2
    onfrac = (1 - gmm_weight)*onfrac_sds + gmm_weight*onfrac_gmm
    aeb = -np.log(1 - onfrac)
    return aeb


def log_transform(flat_data):
    first_percentile = np.quantile(flat_data, 0.01)
    const = first_percentile - 2000
    log_transformed = np.log(flat_data - const)
    return log_transformed


def gaussians(flat_data, flat_len, means_init, reg_covar=REG_COVAR,
        threshold_sds=5):
    mixture = sk.mixture.GaussianMixture(2, means_init=means_init, tol=tol,
        reg_covar=reg_covar, n_init=n_init, max_iter=max_iter,
        covariance_type="spherical")
    try:
        gmm_labels = mixture.fit_predict(flat_data.reshape(-1, 1))
        if reg_covar > 10*REG_COVAR:
            warning_text = ("reg_covar increased to " + str(reg_covar) 
                          + "; should be near " + str(REG_COVAR))
            warnings.warn(warning_text)
        off_label, sds_threshold, ordered_means = mixture_orientation(
            np.ravel(mixture.means_), np.ravel(mixture.covariances_),
            threshold_sds)
        if off_label == 1:
            flat_labels = ((np.ravel(1-gmm_labels)
                    & (flat_data > ordered_means[0]))
                | (flat_data > ordered_means[1]))
        else:
            flat_labels = ((np.ravel(gmm_labels)
                    & (flat_data > ordered_means[0]))
                | (flat_data > ordered_means[1]))
        onfrac_gmm = np.sum(flat_labels) / flat_len
        onfrac_sds = np.sum(flat_data > sds_threshold) / flat_len
        return aeb(onfrac_gmm, onfrac_sds)
    except Exception:
        if reg_covar < 100*REG_COVAR:
            return gaussians(flat_data, flat_len, means_init, reg_covar*2,
                threshold_sds)
        else:
            gmm_labels = mixture.fit_predict(flat_data.reshape(-1, 1))


def mixture_aeb(flat_data, threshold_sds=5):
    flat_len = len(flat_data)
    means_init = np.asarray(
        (np.nanmin(flat_data), np.nanmax(flat_data))).reshape(-1, 1)
    return gaussians(flat_data, flat_len, means_init, REG_COVAR, threshold_sds)


def well_to_aeb(well_entry=None, log=True, threshold_sds=5):
    if isinstance(well_entry, str):
        data = pd.read_csv(well_entry)
    elif hasattr(well_entry, "__iter__"):
        return {entry: well_to_aeb(entry, log=log, threshold_sds=threshold_sds)
                for entry in well_entry}
    elif isinstance(well_entry, os.DirEntry):
        data = pd.read_csv(well_entry.path)
    else:
        well_path = filedialog.askopenfilenames(
            title="Choose a File with the Flow Cytometry Data for One Well",
            filetypes=[("Comma-Separated Values", "csv")], multiple=True)
        return well_to_aeb(well_path)
    flat_data = flatten(data)
    well_aeb = mixture_aeb(flat_data, threshold_sds=threshold_sds)
    return well_aeb


def extended_coefs(concs, aebs, corr="c4", cal_curve=None):
    blank_array = flatten(aebs[concs == 0])
    if cal_curve is not None:
        new_cal_curve = cal_curve
        calibratible = True
    else:
        try:
            new_cal_curve = CalCurve.from_data(concs, aebs, model="4PL")
            calibratible = True
        except Exception:
            calibratible = False
    if calibratible:
        coefs = {key: value for key, value in new_cal_curve.coefs.items()}
        coefs["LOD"] = new_cal_curve.lod
        coefs["LLOQ"] = lod(blank_array, new_cal_curve, sds=10, corr=corr)
    else:
        coefs = {"a": np.nan, "b": np.nan, "c": np.nan, "d": np.nan,
            "LOD": np.inf, "LLOQ": np.inf}
    coefs["blank mean"] = np.mean(blank_array)
    try:
        ddof = {"n": 0, "n-1": 1, "n-1.5": 1.5, "c4": 1}[corr]
    except KeyError:
        ddof = float(corr)
    corr_factor = c4(len(blank_array)) if (corr == "c4") else 1
    coefs["blank std"] = np.std(blank_array, ddof=ddof)/corr_factor
    coefs["blank cv"] = coefs["blank std"] / coefs["blank mean"]
    return coefs


def plate_subsets(dir_path=None, save_aebs_to=None, save_coefs_to=None,
        log: bool = True, model="4PL", lod_sds=3, subsets: int = 10, sizes=(),
        corr: str = "c4", threshold_sds=5):
    plate_files = PlateFileCollection(dir_path)
    concs = []
    datasets = []
    dlens = []
    aebs = []
    for well in plate_files.wells:
        concs.append(plate_files.conc_map[well])
        data = pd.read_csv(plate_files.file_map[well])
        flat_data = log_transform(flatten(data.values))
        dlens.append(len(flat_data))
        datasets.append(flat_data)
        aebs.append(mixture_aeb(flat_data, threshold_sds))
    concs_flat = flatten(concs, on_bad_data="error")
    aebs_flat = flatten(aebs, on_bad_data="error")
    cal_curve = CalCurve.from_data(concs_flat, aebs_flat, model=model,
        lod_sds=lod_sds, corr=corr)
    coefs = extended_coefs(concs_flat, aebs_flat, corr, cal_curve)
    coef_table = pd.DataFrame.from_dict(coefs, orient="index")
    aeb_table = pd.DataFrame.from_dict({
        "Well": plate_files.wells,
        "Concentration": concs_flat,
        "AEB": aebs_flat,
        "Number of Beads": dlens})
    rng = default_rng()
    subset_aebs = {}
    for size in sizes:
        subset_aebs[size] = pd.DataFrame(columns=range(subsets),
            index=plate_files.wells, dtype=float)
        for w in range(len(concs)):
            if dlens[w] > size:
                for subset in range(subsets):
                    subset_data = rng.choice(datasets[w], size, replace=False)
                    subset_aebs[size].at[
                        plate_files.wells[w], subset] = mixture_aeb(
                            subset_data, threshold_sds)
            else:
                for subset in range(subsets):
                    subset_aebs[size].at[
                        plate_files.wells[w], subset] = aebs_flat[w]
        subset_aebs[size]["mean"] = subset_aebs[size][
            range(subsets)].mean(axis=1, skipna=False)
        subset_aebs[size]["std"] = subset_aebs[size][
            range(subsets)].std(axis=1, skipna=False, ddof=1.5)
        subset_aebs[size]["median"] = subset_aebs[size][
            range(subsets)].median(axis=1, skipna=False)
        subset_aebs[size]["IQR 25%"] = subset_aebs[size][
            range(subsets)].quantile(q=0.25, axis=1)
        subset_aebs[size]["IQR 75%"] = subset_aebs[size][
            range(subsets)].quantile(q=0.75, axis=1)
    if save_aebs_to is None:
        save_aebs_to_ = filedialog.asksaveasfilename(
            initialfile=plate_files.name + " subset AEBs.xlsx",
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "xlsx")])
    else:
        save_aebs_to_ = save_aebs_to
    if save_aebs_to_:
        with pd.ExcelWriter(save_aebs_to_) as writer:
            aeb_table.to_excel(writer, sheet_name="Original", index=False)
            for size in sizes:
                subset_aebs[size].to_excel(writer, sheet_name=str(size))
    subset_coefs = {}
    for size in sizes:
        subset_coefs[size] = pd.DataFrame(columns=range(subsets),
            index=coefs_index, dtype=float)
        for subset in range(subsets):
            one_subset_coefs = extended_coefs(
                concs_flat, subset_aebs[size][subset], corr)
            for key, value in one_subset_coefs.items():
                subset_coefs[size].at[key, subset] = value
        subset_coefs[size]["mean"] = subset_coefs[size][
            range(subsets)].fillna(np.inf).mean(axis=1, skipna=False)
        subset_coefs[size]["std"] = subset_coefs[size][
            range(subsets)].fillna(np.inf).std(axis=1, skipna=False, ddof=1.5)
        subset_coefs[size]["median"] = subset_coefs[size][
            range(subsets)].fillna(np.inf).median(axis=1, skipna=False)
        subset_coefs[size]["IQR 25%"] = subset_coefs[size][
            range(subsets)].fillna(np.inf).quantile(q=0.25, axis=1)
        subset_coefs[size]["IQR 75%"] = subset_coefs[size][
            range(subsets)].fillna(np.inf).quantile(q=0.75, axis=1)
    if save_coefs_to is None:
        save_coefs_to_ = filedialog.asksaveasfilename(
            initialfile=plate_files.name + " subset coefficients.xlsx",
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "xlsx")])
    else:
        save_coefs_to_ = save_coefs_to
    if save_coefs_to_:
        with pd.ExcelWriter(save_coefs_to_) as writer:
            coef_table.to_excel(writer, sheet_name="Original", header=False)
            for size in sizes:
                subset_coefs[size].to_excel(writer, sheet_name=str(size))
    return coef_table
