from ckLogging import *

def wraparoundN(values, lower, upper):
    notImplemented("Return of multipliers is not implemented.")
    assert lower < upper, "\'lower\'-value must be lower than \'upper\'-value"

    wrappedValues = values - lower
    upper = upper - lower
    wrappedValues = wrappedValues.real % upper
    wrappedValues = wrappedValues + lower
    return wrappedValues
