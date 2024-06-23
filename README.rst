waltlabtools
============

.. image:: https://img.shields.io/conda/vn/tylerdougan/waltlabtools?label=Anaconda
   :target: https://anaconda.org/tylerdougan/waltlabtools
   :alt: Anaconda
.. image:: https://img.shields.io/pypi/v/waltlabtools?label=PyPI
   :target: https://pypi.org/project/waltlabtools
   :alt: PyPI
.. image:: https://img.shields.io/readthedocs/waltlabtools?label=Documentation
   :target: https://waltlabtools.readthedocs.io/en/latest/
   :alt: Documentation
.. image:: https://img.shields.io/github/issues/tylerdougan/waltlabtools?label=GitHub
   :target: https://github.com/tylerdougan/waltlabtools
   :alt: GitHub

A collection of tools for biomedical research assay analysis in Python.

Key Features
------------

-  Analysis for assays such as
   `digital ELISA <http://www.ncbi.nlm.nih.gov/pmc/articles/pmc2919230/>`__,
   including single-molecule array (Simoa) assays
-  Read instrument-generated files and calculate calibration curves,
   concentrations, limits of detection, and more
-  Free and open-source software under the
   `GNU General Public License v3 <https://www.gnu.org/licenses/gpl-3.0.en.html>`__

Getting Started
---------------

Installation
^^^^^^^^^^^^

You can install waltlabtools using
`Anaconda <https://anaconda.org/tylerdougan/waltlabtools>`__ (recommended) or
`PyPI <https://pypi.org/project/waltlabtools/>`__. If you're not comfortable
with the command line, begin by installing
`Anaconda Navigator <https://www.anaconda.com/products/individual>`__. Then follow
`these instructions <https://docs.anaconda.com/anaconda/navigator/tutorials/manage-channels/>`__
to add the channel ``tylerdougan``, and install waltlabtools from this channel.

Alternatively, install waltlabtools from the command line with
``conda install -c tylerdougan waltlabtools`` (recommended; requires you to
first install Anaconda or
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__) or
``pip install waltlabtools`` (requires
`pip <https://pip.pypa.io/en/stable/getting-started/>`__, which should come
with Python).


Usage
^^^^^

.. code-block:: python
   
   import waltlabtools as wlt  # waltlabtools main functionality

-----


Development of waltlabtools is led by the
`Walt Lab <https://waltlab.bwh.harvard.edu>`__ for Advanced Diagnostics
at `Brigham and Women's Hospital <https://www.brighamandwomens.org>`__,
`Harvard Medical School <https://hms.harvard.edu>`__, and the
`Wyss Institute for Biologically Inspired Engineering <https://wyss.harvard.edu>`__.
