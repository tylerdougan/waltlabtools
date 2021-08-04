# Functions and classes that do not have dependencies.
# (_len_for_message, _error_text, I, isiterable, match_coefs)

def _len_for_message(collection):
    try:
        str_len = str(len(collection))
    except Exception:
        str_len = "an unknown number of"
    return str_len


def _error_text(error_inputs, error_type):
    """
    Return the text for an error.
   
    Parameters:
    ----------
    `error_inputs` : list-like
        The information required to construct the error message. Depends
        on the `error_type`.
    `error_type` : str
        The type of error to be returned. Should be one of:
            - `"length": `error_inputs` should be of the form
            `[provided_collection, desired_collection, plural_title]`.
            - "coercion": `error_inputs` should be of the form
            `[initial_data_type, flattened_data_type]`. 
            - "implementation": `error_inputs` should be of the form
            `[argument_keyword, provided_argument]`. 
   
    Returns
    -------
    `error_text` : str
        An error message.

    """

    if (error_type == "length") and (len(error_inputs) >= 3):
        provided_len = _len_for_message(error_inputs[0])
        desired_len = _len_for_message(error_inputs[1])
        title_str = str(error_inputs[2])
        error_text = ("Incorrect length. Wrong number of " + title_str
            + ". The function requires " + desired_len + " " + title_str 
            + ": " + str(error_inputs[1]) + ". You provided a"
            + str(type(error_inputs[0])) + " with " + provided_len
            + " " + title_str + ": " + str(error_inputs[0]) + ".")
    elif (error_type == "coercion") and (len(error_inputs) >= 2):
        error_text = ("Input data were coerced from type "
            + str(error_inputs[0]) + " to type " + str(error_inputs[1])
            + ", but could not be coerced to an ndarray.")
    elif (error_type == "implementation"):
        error_text = ("The " + str(error_inputs[0]) + " "
            + str(error_inputs[1])
            + " does not exist. Please try another one.")
    else:
        error_text = "An unknown error occurred with " + str(error_inputs)
    return error_text


# GENERAL FUNCTIONS
def Id(x):
    """
    The identity function.
   
    Parameters:
    ----------
    `x` : any
   
    Returns
    -------
    `x` : same as `x`
        The same `x`, unchanged.
       
    """
    return x


def isiterable(data):
    """
    Determines whether an object is iterable, as determined by it
    having a nonzero length and not being a string.
   
    Parameters:
    ----------
    `data` : any
   
    Returns
    -------
    `iterable` : bool
        Returns `True` iff `data` is not a string, has `len`, and
        `len(data) > 0`.
       
    """
    try:
        return ((len(data) > 0) and not isinstance(data, str))
    except Exception:
        return False


def _match_coefs(params, coefs):
    if (isinstance(coefs, dict) 
            and (set(coefs.keys()) == set(params))):
        coefs_dict = coefs
    elif isiterable(coefs):
        if len(coefs) == len(params):
            coefs_dict = {params[i]: coefs[i] for i in range(len(coefs))}
        else:
            raise ValueError(_error_text([coefs, params], "length"))
    elif len(params) == 1:
        coefs_dict = {params[0]: coefs}
    elif (not params) and (not coefs):
        coefs_dict = {}
    else:
        raise ValueError(_error_text([coefs, params], "length"))
    return coefs_dict
