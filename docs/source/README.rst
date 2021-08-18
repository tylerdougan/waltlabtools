waltlabtools
============

A collection of tools for biomedical research assay analysis in Python.

Key Features
------------

-  Analysis for assays such as `digital
   ELISA <https://dx.doi.org/10.1038%2Fnbt.1641>`__.
-  Calculation of calibration curves, concentrations, limits of
   detection, and more.
-  Free and open-source software under the `GNU General Public License
   v3 <https://www.gnu.org/licenses/gpl-3.0.en.html>`__.

Installation
------------

-  Download **waltlabtools** from
   `PyPI <https://pypi.org/project/waltlabtools/>`__.
-  Dependencies: **waltlabtools** requires
   `scipy <https://docs.scipy.org/doc/scipy/getting_started.html>`__ ≥
   1.3.1, and either `jax <https://jax.readthedocs.io/en/latest/>`__ ≥
   0.1.64 or `numpy <https://numpy.org/doc/stable/index.html>`__ ≥
   1.10.0. To make the best use of **waltlabtools**, you may want to
   install `pandas <https://pandas.pydata.org>`__ (for data
   import/export and organization),
   `matplotlib <https://matplotlib.org>`__ (for plotting), and
   `JupyterLab <https://jupyterlab.readthedocs.io/en/stable/>`__ (for
   writing code). These can all be installed using
   `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`__,
   and may become dependencies in future releases.

Functions and Classes
---------------------

.. include:: /generated/waltlabtools.core.rst
   :start-line: 3

.. include:: /generated/waltlabtools.nonnumeric.rst
   :start-line: 3

-----

Development of **waltlabtools** is led by the `Walt
Lab <https://waltlab.bwh.harvard.edu>`__ for Advanced Diagnostics at
Brigham and Women's Hospital, Harvard Medical School, and the Wyss
Institute.
