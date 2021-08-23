"""
:noindex:

"""

# Functions and classes that do not have dependencies.
# (_len_for_message, _error_text, I, isiterable, match_coefs)


def _len_for_message(collection):
    try:
        str_len = str(len(collection))
    except Exception:
        str_len = "an unknown number of"
    return str_len


def _error_text(error_inputs, error_type):
    """Text for an error.
   
    Parameters
    ----------
    error_inputs : list-like
        The information required to construct the error message. Depends
        on the **error_type**.
    error_type : ``str``
        The type of error to be returned. Should be one of:

            - "length": **error_inputs** should be of the form
              `[provided_collection, desired_collection, plural_title]`.

            - "coercion": **error_inputs** should be of the form
              `[initial_data_type, flattened_data_type]`. 

            - "implementation": **error_inputs** should be of the
              form `[argument_keyword, provided_argument]`.

            - "nonpositive": **error_inputs** should be of the form
              `[nonpositive_values]`.

            - "_match_model": **error_inputs** should be of the form
              `[m]` or `[m, matches]`.
   
    Returns
    -------
    error_text : ``str``
        An error message.

    """
    if (error_type == "length") and (len(error_inputs) >= 3):
        provided_len = _len_for_message(error_inputs[0])
        desired_len = _len_for_message(error_inputs[1])
        title_str = str(error_inputs[2])
        error_text = ("Incorrect input size. Wrong number of " + title_str
            + ". The function requires " + desired_len + " " + title_str 
            + ": " + str(error_inputs[1]) + ". You provided a"
            + str(type(error_inputs[0])) + " with " + provided_len
            + " " + title_str + ": " + str(error_inputs[0]) + ".")
    elif (error_type == "coercion") and (len(error_inputs) >= 2):
        error_text = ("Input data were coerced from type "
            + str(error_inputs[0]) + " to type " + str(error_inputs[1])
            + ", but could not be coerced to an ndarray.")
    elif (error_type == "implementation") and (len(error_inputs) >= 2):
        error_text = (str(error_inputs[0]) + " is not a valid "
            + str(error_inputs[1]) + ". Please try another one.")
    elif (error_type == "nonpositive") and (len(error_inputs) >= 1):
        error_text = ("Geometric mean requires all numbers to be nonnegative. "
            + "Because the data provided included " + str(error_inputs[0])
            + ", the geometric meandian is unlikely to provide any insight.")
    elif (error_type == "_match_model") and (len(error_inputs) >= 1):
        error_text = "Model " + error_inputs[0] + " not found."
        if len(error_inputs) >= 2:
            error_text = (error_text
                + " Did you mean one of " + str(error_inputs[1]) + "?")
    else:
        error_text = "An unknown error occurred with " + str(error_inputs)
    return error_text


# GENERAL FUNCTIONS
def Id(x):
    """The identity function.
    
    Parameters
    ----------
    x : any
    
    Returns
    -------
    x : same as input
        The same **x**, unchanged.
    
    """
    return x


def _match_coefs(params, coefs):
    if isinstance(coefs, dict):
        if set(coefs.keys()) == set(params):
            coefs_dict = coefs
        else:
            raise ValueError(_error_text([coefs, params], "length"))
    elif hasattr(coefs, "__iter__"):
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
