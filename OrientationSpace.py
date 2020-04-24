import collections

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import fftpack

from ckLogging import notImplemented


def angularKernel(K=5, angles=np.array([]), N=1024):
    n = 2*K + 1
    if not np.any(angles) :
        angles = np.arange(n)*(np.pi/n);
    if isinstance(N, collections.Mapping):
        coords = N
    else:
        coords = getFrequencySpaceCoordinates(N)
    coords["t"] = coords["t"][:, :, None] - angles[None, None, :]
    coords["t"]  = ( (coords["t"]+np.pi) % (2*np.pi) ) - np.pi

    # % % Angular part
    # % scale the angle
    s_a = np.pi / (2*K + 1)
    theta_s = coords["t"] / s_a;
    angularFilter = 2*np.exp( -(theta_s**2) /2 )
    angularFilter_shifted = angularFilter[ ::-1, ::-1, :]
    filterKernel = 0.5 * (angularFilter + angularFilter_shifted)
    # % could we simplify this? neg/pos dividing line does not have to rotate
    posMask = np.abs(coords["t"]) < np.pi/2
    filterKernel = filterKernel*(1 + 1j*(posMask*2-1))
    return filterKernel

def radialKernel(f_c, b_f=None, N=1024):
    if b_f == None:
        b_f = f_c/np.sqrt(2)
    if isinstance(N, collections.Mapping):
        coords = N
    else:
        coords = getFrequencySpaceCoordinates(N)

    # %% Radial part
    # % compute radial order, f_c = sqrt(K_f * b_f^2)
    if f_c != None:
        K_f = (f_c / b_f) **2;
        # % scale frequency
        f_s = coords["f"]/f_c

        """
        # % Equation 3.11
        # % Note -(f ^ 2 - f_c ^ 2) / (2 * b_f ^ 2)
        # %   = (1 - (f / f_c) ^ 2) / (2 * b_f ^ 2 / f_c ^ 2)
        # %   = (1 - (f / f_c) ^ 2) / (2 / K_f)
        """

        radialFilter = f_s**K_f
        radialFilter = radialFilter*np.exp( (1-f_s**2)*(K_f/2) )

    else:
        radialFilter = np.exp( - ( (coords["f"]**2) / (2*b_f**2) ) )

    return radialFilter



"""



    % Equation 3.11
    % Note -(f^2 - f_c^2)/(2*b_f^2) = (1 - (f/f_c)^2)/(2* b_f^2/f_c^2)
    %                               = (1 - (f/f_c)^2)/(2 / K_f)
    % radialFilter = f_s.^K_f .* exp((1 - f_s.^2)*K_f/2);
    radialFilter = bsxfun(@power,f_s,K_f);
    radialFilter = radialFilter .* exp(bsxfun(@times,(1 - f_s.^2),K_f/2));
    % radialFilter2 = f_s^K_f .* exp(-(f.^2-f_c.^2)/2/b_f.^2);
    % assertEqual(radialFilter,radialFilter2);
else
    radialFilter = exp(-bsxfun(@rdivide,coords.f.^2,2*b_f.^2));
    radialFilter(1) = 0;
end

end
"""
#    return radialFilter



def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho   = np.sqrt(x**2 + y**2)
    return (theta, rho)

def getFrequencySpaceCoordinates(N=1024):
    """
    :param N:
    :return: [cartesian, polar]
    """

    notImplemented("persistent lastN lastCartesian lastPolar")
    if np.isscalar(N):
        N = np.array([N, N])

    coords = {}
    coords["x"], coords["y"] = np.meshgrid(
                                    np.arange(0, N[1]) - np.floor(N[1] / 2),
                                    np.arange(0, N[0]) - np.floor(N[0] / 2)
                                                )

    #coords["x"] = np.fft.ifftshift(coords["x"])
    #coords["y"] = np.fft.ifftshift(coords["y"])

    coords["x"] = fftpack.ifftshift(coords["x"])
    coords["y"] = fftpack.ifftshift(coords["y"])





    coords["t"], coords["f"] = cart2pol( coords["x"] / np.floor(N[1] / 2) / 2,
                                       coords["y"] / np.floor(N[0] / 2) / 2
                                     )
    return( coords )


def kernel(fc, bf, K, angle, N):
    """ Still gotta figure out what's happening here exactly. """
    pass


def blendOrientationMap( theta, res=np.array([]), cm=cm.get_cmap("hsv") ):
    #raise NotImplementedError("I stopped implementing blendOrientationMap, "
    #                          "because it looks useless.")
    if res.size == 0:
        res = np.ones(theta.shape)

    theta_not_nan = np.isnan(theta)

    scaled = theta / np.pi

    theta_mapped = cm(scaled)[:,:,0:3]
    theta_mapped[theta_not_nan] = np.NaN

    res = res.real
    res[res<0] = np.NaN
    res = res / np.nanmax(res)

    blended_map = theta_mapped * res[:,:,None]

    return blended_map












