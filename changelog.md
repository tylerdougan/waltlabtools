0.2.13
 - Defaults to numpy backend unless jax has already been imported.
 - backend.py has been renamed to nonnumeric.py to clarify that it is simply the functions which do not require any additional packages.
 - Docstrings have been reformatted to comply better with numpydoc.
 - Added `gmnd` function for the geometric meandian.

0.2.14
 - bug fix

0.2.31
 - renamed `waltlabtools` module to `core`

0.2.32
 - ReadTheDocs test
 
0.2.40
 - Added type hints

0.3.0
 - Added modules read_hdx and mosaic
 
0.3.4
 - Added docs_requires.txt

0.3.5
 - Removed unnecessary tkinter requirement

0.3.6
 - Removed some type hints and improved documentation

0.3.7
 - Renamed read_hdx module to read_quanterix to provide for future expansion to SP-X and other instrumentation
 - Added module docstrings
 - Attempted to build conda packages for different versions of conda
 - Added read_quanterix.sample_results function

0.3.8
 - Fixed a bug in read_quanterix

0.3.9
 - Rearranged modules