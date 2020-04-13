
from collections import UserDict
import numpy as np

from imageSegmentation import *

class DefaultParams(UserDict):
    """ Based on arguments of 
    steerableAdaptiveResolutionOrientationSpaceDetector.m
    Comments copied from Kittisopikul's code. 
    
    order - (optional)
        K_h parameter that determines the highest *K* value used for initial
        filtering via OrientationSpaceFilter
        Type: Numeric, scalar
        Default: 8
    
    sigma- (optional)
        scale parameter setting the radial bandpass in pixels
        central frequency, f_c, of the bandpass filter will be 1/(2*pi*sigma)
        Type: Numeric, scalar
        Default: 2 (pixels)
    
    ADVANCED INPUTS (NAMED PARAMETERS)
    
    adaptLengthInRegime - Adapt the resolution with the highest regime by 
        searching for the maxima with the smallest derivative with respect to
        *K*;
        Type: logical
        Default: true
    
    meanThresholdMethod - Function to determine threshold of mean response
        Type: char, function_handle
        Default: @thresholdOtsu
    
    mask - Binary mask the same size as I to limit the area of processing
        Type: logical
        Default: []

    nlmsMask - Override mask for NLMS processing. N x M
        Type: logical
        Default: [] (Calculate mask using mean filter response)
       
    nlmsThreshold - Override attenuated mean response threshold to apply to
        NLMS
        Type: numeric, 2D
        Default: [] (Use AMOR)
    
    useParallelPool - Logical if parallel pool should be used
        Type: logical
        Default: true
    
    maskDilationDiskRadius - Disc structure element radius in pixels to dilate
        the mask calculated from the mean response
        Type: numeric
        Default: 3
    
    maskFillHoles - Logical indicating if holes should be filled in the
        nlmsMask. True indicates to holes should be filled.
        Type: logical
        Default: false
    
    diagnosticMode - True if diagnostic figures should be shown
        Type: logical, scalar
        Default: false
    
    K_sampling_delta - Interval to sample K when using adaptLengthInRegime
        Type: numeric, scalar
        Default: 0.1
    
    responseOrder - K_m, orientation filter resolution at which to calculate
        the response values;
        Type: numeric, scalar
        Default: 3
    
    bridgingLevels - Number of bridging steps to complete. A value of 1 or 2
        is valid.
        type: numeric, scalar
        Default: 2
    
    suppressionValue - Value to assign to pixels that are suppressed in the
        NMS/NLMS steps
        Type: numeric, scalar
        Default: 0
    
    filter - OrientationSpaceFilter object instance to use, overrides order
        and sigma parameters; Used to share filter initialization between many
        function calls
        Type: OrientationSpaceFilter
        Default: Create new filter based on order and sigma inputs
    
    response - OrientationSpaceResponse object to use, overrides order, sigma,
        and filter; used to share filter response between many function calls.
        Type: OrientationSpaceResponse
        Default: Convolve filter with the response to calculate the response

    ### FURTHER ADVANCED INPUTS: UNSERIALIZATION INPUTS (NAMED PARAMETERS)
    These parameters allow some of the output in the struct *other*, below,
    to be fed back into the function in order to obtain the full output of the
    function. The purpose of this is so that the full output can be
    regenerated from a subset of the output that has been saved to disk, or
    otherwise serialized, without the need for complete re-computation.

    maxima_highest - numeric 3D array
    K_highest - numeric 3D array
    bridging - struct array
    nlms_highest - numeric 3D array
    nlms_single - numeric 2D array
    
    """

    def __init__(self):
        UserDict.__init__(self)
        self.update(
            {
            "order": 8,
            "sigma": 2,

            "adaptLengthInRegime":      True,
            "meanThresholdMethod":      thresholdOtsu,
            "meanThreshold":            None,
            "mask":                     np.empty(0),
            "nlmsMask":                 np.empty(0),
            "nlmsThreshold":            np.empty(0),
            "useParallelPool":          True,
            "maskDilationDiskRadius":   3,
            "maskFillHoles":            False,
            "diagnosticMode":           True,
            "K_sampling_delta":         0.1,
            "responseOrder":            3,
            "bridgingLevels":           2,
            "suppressionValue":         0,
            "filter":                   False, #OrientationSpaceFilter(),
            "response":                 False, #OrientationSpaceResponse(),

            "maxima_highest":   np.empty(0),
            "K_highest":        np.empty(0),
            "bridging":         np.empty(0),
            "nlms_highest":     np.empty(0),
            "nlms_single":      np.empty(0),
            }
        )

if __name__ == "__main__":
    ip = DefaultParams()
    print(type(ip))
    print(ip)
    for key, value in ip.items():
        print(key, value)

