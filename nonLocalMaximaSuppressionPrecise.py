import os

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import *
from scipy.signal import resample
from scipy.interpolate import interpn, RegularGridInterpolator, LinearNDInterpolator

from skimage.util import montage
from skimage.io   import imsave
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
        offsetAngle[offsetAngle==np.NaN] = np.nanmean(theta)

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
        #angleIdx = angleIdx / np.pi * period + 2;
        angleIdx = angleIdx / np.pi * period +1;
        # % Negative values should be advanced by one period
        # % angleIdx(angleIdx < 0) = angleIdx(angleIdx < 0)+period;

    # % Offset by 1 due to padding
    #x, y = np.meshgrid( np.arange(2, nx+1+1), np.arange(2, ny+1+1) )

    if True:
        # first try, caused weird problems in interpolations
        x, y = np.meshgrid(np.arange(1,nx+1), np.arange(1,ny+1))
    else:
        # attempt to solve the problem,
        # must be flipped for scipy.interpolate.interpn for some reason
        x, y = np.meshgrid(np.arange(nx,0,-1), np.arange(ny, 0, -1))


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

    # must be flipped for scipy.interpolate.interpn for some reason
    # or not: angleIdx = angleIdx*-1 + np.nanmin(angleIdx) + np.nanmax(angleIdx)

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

    print(x.shape)
    print(y.shape)
    print(angleIdx.shape)
    print(rotationResponse.shape)



    x_ = np.arange(rotationResponse.shape[0])
    y_ = np.arange(rotationResponse.shape[1])
    z_ = np.arange(rotationResponse.shape[2])
    print( x_.shape )
    print( y_.shape )
    print( z_.shape )

    print(np.nanmin(x), np.nanmax(x))
    print(np.nanmin(y), np.nanmax(y))
    print(np.nanmin(angleIdx), np.nanmax(angleIdx))


    if angleMultiplier != 1:
        notImplemented("Cubic 3D interpolation not available in Python, using linear.")
        interpMethod = "linear"
        if True:
            if True:
                #rotationResponse = np.ones(rotationResponse.shape) * np.arange(np.flip(rotationResponse.shape[2]))[None,None,:]
                interpolator = RegularGridInterpolator( (x_, y_, z_),
                                                        rotationResponse,
                                                        method = interpMethod,
                                                        fill_value = 0,
                                                        bounds_error=False)
            else:
                interpolator = LinearNDInterpolator( (x_, y_, z_),
                                                     rotationResponse,
                                                     fill_value = 0)

            """            
            x, y = np.meshgrid( np.arange(512), np.arange(512) )
            x = np.tile(x[:,:,None], 53 )
            y = np.tile(y[:,:,None], 53)
            x = np.expand_dims(x, 2)
            y = np.expand_dims(y, 2)

            angleIdx = np.ones( (512,512) )[:,:,None] * np.arange(53)[None, None, :]
            angleIdx = np.expand_dims(angleIdx, 2)"""

            #print(x.shape, y.shape, angleIdx.shape)


            A = interpolator( (y, x, angleIdx) )


            #angleIdx = np.ones((512,512, 1,53))*np.arange(53)[None,None, None,:]
            for i in range(0, A.shape[-1]):
                #angleIdx = angleIdx*-1 + np.nanmin(angleIdx) + np.nanmax(angleIdx)
                #points = np.array([x[:,:,0,i], y[:,:,0,i], angleIdx[:,:,0,i]])
                #points = np.moveaxis(points, 0, 2)
                #print(points.shape)
                #A = interpolator(points)
                #A = interpolator( (x[:,:,0,i], y[:,:,0,i], angleIdx[:,:,0,i]) )
                imsave(os.path.join("outputImages/pyAreg", "pyAiter_%d.tif"%i),
                       A[:,:,0,i].astype("single"))

        else:
            #angleIdx = np.ones((512,512,1,3))
            A = interpn(
                (y_, x_, z_),
                rotationResponse,
                #(x, y, angleIdx),
                (x[:,:,0,1], y[:,:,0,1], angleIdx[:,:,0,1]),
                method=interpMethod,
                fill_value=0,
                bounds_error=False)

            #imsave(os.path.join("outputImages/pyA", "pyA_single1.tif"),A.astype("single"))

            print(A.shape)
            for i in range(0, A.shape[-1]):
                imsave(os.path.join("outputImages/pyA", "pyA_%d.tif"%i),
                       A[:,:,0,i].astype("single"))

        for i in range(0, x.shape[3]):
            imsave(os.path.join("outputImages/pyx", "pyx_%d.tif" % i),
                   x[:, :, 0, i].astype("single"))

        for i in range(0, y.shape[3]):
            imsave(os.path.join("outputImages/pyy", "pyy_%d.tif" % i),
                   y[:, :, 0, i].astype("single"))

        for i in range(0, angleIdx.shape[3]):
            imsave(os.path.join("outputImages/pyangleIdx", "pyangleIdx_%d.tif" % i),
                   angleIdx[:, :, 0, i].astype("single"))

        for i in range(0, rotationResponse.shape[2]):
            imsave(os.path.join("outputImages/pyRR", "pyRRx_%d.tif" % i),
                   rotationResponse[:, :, i].astype("single"))