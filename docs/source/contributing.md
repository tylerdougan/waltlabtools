Contributing to waltlabtools
============================
We welcome contributions to waltlabtools from anyone. These can include
bug fixes, new features, code improvement, examples, tests, and
documentation. Although the full scope of waltlabtools is not yet
circumscribed, it does include the following:
  - functions for interfacing with lab equipment such as plate readers,
    flow cytometers, and
    [Quanterix](https://www.quanterix.com/products-and-services/)
    instruments
  - wrappers for repeatable data analysis workflows using, _e.g._,
    [scikit-learn](https://scikit-learn.org/stable/), especially
    classifiers for diagnosis of disease
  - scripts for automating liquid handling (calculations or robotics)
    for reagent preparation or sample processing
  - scripts to automate repetitive tasks in research, such as 
    converting between mass and molar concentrations or downloading
    papers

Submitting contributions
------------------------
Contributions (including new features, feature requests, documentation,
examples, improvements to existing code, and bug reports) may be
submitted one of two ways.

1. On GitHub: To highlight a bug or shortcoming, file a [bug report]. To
   submit an improvement, fork the repository on GitHub, add your
   changes, and submit a pull request.

2. By email to [tyler_dougan@hst.harvard.edu].

GitHub bug reports and pull requests are highly recommended for all
submissions because they allows each author's contributions to be
acknowledged. Email submission is provided as an alternative route for
those who want to contribute a compartmentalized function or module
without using git. Email submission may be deprecated in the future.

Code guidelines
---------------

### Linting
All code should conform to [PEP8] guidelines, with the sole exception that
the maximum code line length is 88 characters. The best way to check
this is to use [flake8] with `--max-line-length=88`.

### Formatting
Format your code using [black].

### Documentation
All docstrings should conform to the [numpydoc] standards. Other
formatted text pages (e.g., README) should be written in
[ReStructuredText]. Use [pandoc] to convert from Markdown and other
formats to ReStructuredText. All examples should be written as
JupyterLab notebooks.

### Compatibility
The following rules can be broken if necessary, with permission.
  - Code should be compatible with Python 3.7, 3.8, 3.9, and 3.10
  - Code should work equally well using [numpy] or [jax.numpy]
  - Code compiled [just-in-time (JIT)] should not raise any different
    errors or return different results when using [numba.jit],
    [jax.jit], or no JIT
