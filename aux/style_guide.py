# Examples

def my_function(a, b, c=True, d=None):
    """Noun phrase or 3rd person present active indicative, one line.
    
    Longer description, 3rd person present active indicative. Around 2-5
    lines. Can be ommitted if one-line description is sufficient.
    
    Parameters
    ----------
    a : type
        Description of parameter `a`. Specify type here, not with type
        hints.
    b
        Description of parameter `b` (with type not specified).
    c : bool, optional
        Description of parameter `c`. Defaults to True, which implies
        whatever `c=True` implies.
    d : type, optional
        Description of parameter `d`.

    Returns
    -------
    type
        Description of return value. Can also specify both the name and
        type of the return value in the same form as the Parameters
        section.

    Raises
    ------
    ExceptionType
        If something bad is passed. Only non-obvious or common 'gotcha'
        errors need to be documented.

    Warns
    -----
    WarningType
        If something worrying is passed. Similar to 'Raises' section.

   """
   return None


# Tyler
def MyClass(a):
    """Noun phrase or 3rd person present active indicative, one line.
    
    Longer description, 3rd person present active indicative. Around 2-5
    lines. Can be ommitted if one-line description is sufficient.
    
    Parameters
    ----------
    a : type
        Description of parameter `a`. Specify type in the docstring, not
        with type hints.

    Attributes
    ----------
    e : type
        Description of attribute `e`.

    Methods
    -------
    my_method(f)
        Description of method `class_method` which takes parameter `f`.
        Only public methods need to be documented here. Methods should
        also be documented like functions where they are defined inside
        the class. Do not include `self` in the list of parameters.

   """
