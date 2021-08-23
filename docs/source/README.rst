waltlabtools
============

.. image:: https://anaconda.org/tylerdougan/waltlabtools/badges/platforms.svg
   :target: https://anaconda.org/tylerdougan/waltlabtools
   :alt: Platforms
.. image:: https://anaconda.org/tylerdougan/waltlabtools/badges/downloads.svg
   :target: https://anaconda.org/tylerdougan/waltlabtools
   :alt: downloads
A collection of tools for biomedical research assay analysis in Python.

Key Features
------------

-  Analysis for assays such as
   `digital ELISA <https://dx.doi.org/10.1038%2Fnbt.1641>`__.
-  Calculation of calibration curves, concentrations, limits of
   detection, and more.
-  Free and open-source software under the
   `GNU General Public License v3 <https://www.gnu.org/licenses/gpl-3.0.en.html>`__.

Getting Started
---------------

-  Installation: ``waltlabtools`` can be installed using
   `conda <https://anaconda.org/tylerdougan/waltlabtools>`__ or
   `pip <https://pypi.org/project/waltlabtools/>`__. In the command
   line,

   - conda: ``conda install -c tylerdougan waltlabtools``

   - pip: ``pip install waltlabtools``

-  Dependencies: ``waltlabtools`` requires
   `scipy <https://docs.scipy.org/doc/scipy/getting_started.html>`__ ≥
   1.3.1, and either `jax <https://jax.readthedocs.io/en/latest/>`__ ≥
   0.1.64 or `numpy <https://numpy.org/doc/stable/index.html>`__ ≥
   1.10.0. To make the best use of ``waltlabtools``, you may want to
   install `pandas <https://pandas.pydata.org>`__ (for data
   import/export and organization),
   `scikit-learn <https://scikit-learn.org/stable/>`__ (for data
   analysis), `matplotlib <https://matplotlib.org>`__ (for plotting),
   and `JupyterLab <https://jupyterlab.readthedocs.io/en/stable/>`__
   (for writing code). These can all be installed using
   `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`__
   or `pip <https://pypi.org>`__, and may become dependencies in future
   releases.

Functions and Classes
---------------------

API: ``waltlabtools`` includes classes for mathematical models and
calibration curves, and a set of functions to make use of these objects
and others. These are covered in the
`documentation <https://waltlabtools.readthedocs.io/README.html>`__.

.. include:: /generated/waltlabtools.core.rst
   :start-line: 2
   :alt: Hello World

.. include:: /generated/waltlabtools.nonnumeric.rst
   :start-line: 2


-----


Development of ``waltlabtools`` is led by the
`Walt Lab <https://waltlab.bwh.harvard.edu>`__ for Advanced Diagnostics
at `Brigham and Women's Hospital <https://www.brighamandwomens.org>`__,
`Harvard Medical School <https://hms.harvard.edu>`__, and the
`Wyss Institute for Biologically Inspired Engineering <https://wyss.harvard.edu>`__.
