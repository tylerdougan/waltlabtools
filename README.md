# `waltlabtools`
A collection of tools for biomedical research assay analysis in Python.


## Key Features
- Analysis for assays such as [digital ELISA](https://dx.doi.org/10.1038%2Fnbt.1641).
- Calculation of calibration curves, concentrations, limits of detection, and more.
- Free and open-source software under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html).


## Installation
- When **waltlabtools** is released, it will be available to download from [PyPI](https://packaging.python.org/tutorials/packaging-projects/) and [conda](https://conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs-skeleton.html).
- Dependencies: **waltlabtools** requires [scipy](https://docs.scipy.org/doc/scipy/getting_started.html) ≥ 1.4.0, and either [jax](https://jax.readthedocs.io/en/latest/) ≥ 0.1.64 or [numpy](https://numpy.org/doc/stable/index.html) ≥ 1.12. To make the best use of **waltlabtools**, you may want to install [pandas](https://pandas.pydata.org) (for data import/export and organization), [matplotlib](https://matplotlib.org) (for plotting), and [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) (for writing code). These can all be installed using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html).
- For development and testing of the pre-release version, download [src/waltlabtools/waltlabtools.py](src/waltlabtools/waltlabtools.py) and put it in Python's [search path](https://docs.python.org/3/tutorial/modules.html#the-module-search-path). One simple option is to put this at the top of your code:
  ```python
  import sys
  sys.path.append("/path/to/waltlabtools")
  import waltlabtools as wlt
  ```

##

Development of **waltlabtools** is led by the [Walt Lab](https://waltlab.bwh.harvard.edu) for Advanced Diagnostics at Brigham and Women's Hospital, Harvard Medical School, and the Wyss Institute.
