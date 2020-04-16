
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import *
from scipy.signal import resample
from scipy.interpolate import RegularGridInterpolator

from skimage.util import montage

from ckLogging import *

def nonLocalMaximaSuppressionPrecise(rotationResponse,
                                     theta,
                                     suppressionValue   = 0,
                                     interpMethod       = "linear",
                                     mask               = np.array([]),
                                     offsetAngle        = None,
                                     angleMultiplier    = 3
                                     ):
    if len(theta.shape) < 3:
        nO = 1
    else:
        # Does this really work with anything else than nO=1 ?
        nO = theta.shape[2]

    if mask.size != 0 :
        mask = binary_dilation(mask, selem=disk(3))

    if offsetAngle == None:
        offsetAngle = theta

    rotationResponseSize = rotationResponse.shape
    if rotationResponse.ndim < 3 :
        rotationResponseSize[2] = 1

    ny = rotationResponseSize[0]
    nx = rotationResponseSize[1]
    nAngles = rotationResponseSize[2]
    period = nAngles * angleMultiplier

    # % See Boyd, Chebyshev and Fourier Spectral Methods, Second Edition
    # % (Revised), Dover, 2001. Toronto. ISBN 0-486-41183-4, page 198
    if mask.size == 0:
        if angleMultiplier != 1:
            rotationResponse = resample(rotationResponse,period, axis=2);

    else:
        rotationResponse = np.moveaxis(rotationResponse, 2, 0)
        rotationResponseTemp = resample(rotationResponse[:,mask],period, axis=0)
        rotationResponse = np.zeros( (period, mask.shape[0], mask.shape[1]))
        rotationResponse[:,:,:] = np.NaN
        rotationResponse[:,mask] = rotationResponseTemp;
        rotationResponse = np.moveaxis(rotationResponse, 0, 2)
        rotationResponseTemp = 0 #clear rotationResponseTemp;

    rotationResponse = np.pad(rotationResponse, ((1,1), (1,1), (0,0)), 'symmetric');
    if angleMultiplier != 1:
        rotationResponse = np.pad(rotationResponse, ((0,0), (0,0), (1,1)), 'wrap');

    # % Map angles in from [-pi/2:pi/2) to (0:pi]
    angleIdx = theta

    if angleMultiplier != 1:
        angleIdx[angleIdx < 0] = angleIdx[angleIdx < 0] + np.pi;
        # % Map angles in radians (0:pi) to angle index (2:nAngles*3+1)
        angleIdx = angleIdx / np.pi * period + 2;
        # % Negative values should be advanced by one period
        # % angleIdx(angleIdx < 0) = angleIdx(angleIdx < 0)+period;

    # % Offset by 1 due to padding
    x, y = np.meshgrid( np.arange(2, nx+1+1), np.arange(2, ny+1+1) )

    x_offset = np.cos(offsetAngle)
    y_offset = np.sin(offsetAngle)

    Xplus  = x + x_offset
    Yplus  = y + y_offset

    Xminus = x - x_offset
    Yminus = y - y_offset

    notImplemented("Return of offset, X, Y from nonLocalMaximaSuppressionPrecise "
                   "not implemented.")
    """
    if(nargout > 1)
        % Extra Chebfun points
        m = sqrt(2)/2;
        XplusCheb = bsxfun(@plus,x,x_offset.*m);
        YplusCheb = bsxfun(@plus,y,y_offset.*m);
        
        XminusCheb = bsxfun(@minus,x,x_offset.*m);
        YminusCheb = bsxfun(@minus,y,y_offset.*m);
    end
    """


    Xstack = np.tile( x[:,:,None], nO)
    Ystack = np.tile( y[:,:,None], nO)

    x = np.block( [ Xminus[:,:,None], Xstack, Xplus[:,:,None] ] )
    y = np.block( [ Yminus[:,:,None], Ystack, Yplus[:,:,None] ] )
    x = np.expand_dims(x, 2)
    y = np.expand_dims(y, 2)

    angleIdx = np.tile(angleIdx[:,:,None, None], 3 )

    """
    if(nargout > 1)
        % Extra Chebfun points
        x = cat(4,x,XplusCheb,XminusCheb);
        y = cat(4,y,YplusCheb,YminusCheb);
        %     if(angleMultiplier ~= 1)
                angleIdx(:,:,:,4:5) = angleIdx(:,:,:,1:2);
        %     end        
        clear XplusCheb YplusCheb XminusCheb YminusCheb        
        end
    """
    Xminus  = 0
    Xstack  = 0
    Xplus   = 0
    Yminus  = 0
    Ystack  = 0
    Yplus   = 0
    x_offset = 0
    y_offset = 0

    print(x)
    if angleMultiplier != 1:
        notImplemented("Cubic 3D interpolation not available in Python, using linear.")
        A = RegularGridInterpolator((x[:,:,0], y[:,:,0], angleIdx[:,:,0]),
                                    rotationResponse,
                                    method = interpMethod,
                                    fill_value = 0)
        print(A.shape)
