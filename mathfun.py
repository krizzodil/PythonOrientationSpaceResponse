import numpy as np
import logging
from scipy import fftpack
from skimage import io
import matplotlib.pyplot as plt
from ckLogging import *

def wraparoundN(values, lower, upper):
    notImplemented("Return of multipliers is not implemented.")
    assert lower < upper, "\'lower\'-value must be lower than \'upper\'-value"

    wrappedValues = values - lower
    upper = upper - lower
    wrappedValues = wrappedValues.real % upper
    wrappedValues = wrappedValues + lower
    return wrappedValues


def isvector(arr):
    return np.prod(arr.shape) == max(arr.shape)

def interpft1(x=None, v= None, xq=None, method="horner", fineGridFactor=None, legacy=False):
    """ Careful: arguments might be slightly different than matlab original
        [v, xq]  = varargin{:};
        [v, xq, method] = varargin{:};
        [x, v, xq] = varargin{:};
        [x,v,xq,method] = varargin{:}; <----
        [v,xq,method,fineGridFactor] = varargin{:};
        [x, v,xq,method,fineGridFactor] = varargin{:};
        [x, v,xq,method,fineGridFactor,legacy] = varargin{:};
    """

    if isvector(v):
        logging.info(("mathfun.interpft1() v.flatten()"))
        v.flatten()
    if isvector(xq):
        logging.info(("mathfun.interpft1() xq.flatten()"))
        xq.flatten()

    if x.size == 0 or np.all(x == None):
        x = np.array([1, v.shape[0]+1])
    elif x.size != 2:
        raise ValueError("x must contain zero or 2 elements.")

    if legacy == False:
        notImplemented("Assertions in interpft1 not implemented!")

    if legacy:
        raise NotImplementedError("I think i didn't implement the legacy-mode "
                                  "of mathfun.interpft1().")
        if isvector(v):
            vSz = [];
        else:
            vSz = v.shape

        if isvector(xq) and not isvector(v):
            outSz = xq.shape[1]
        else:
            outSz = xq.shape

        outSz = np.array([outSz, vSz[1:]])
        xq = xq.flatten()

    # % Map indices from [x(1) x(2)) to [0 1)
    period = x[1] - x[0]
    xq = xq - x[0]
    xq = np.mod(xq.real, period) + 1j * xq.imag
    xq = xq / period

    done = True

    if method=="horner":
        # % domain [0,2)
        xq = xq*2
        if np.all(np.logical_or(np.isreal(xq), np.isnan(xq))):
            logging.info("Mathfun.interpft1() "
                         "np.all(np.logical_or(np.isreal(xq), np.isnan(xq))==True")
            vq = horner_vec_real(v,xq);
        else:
            logging.info("Mathfun.interpft1() "
                         "np.all(np.logical_or(np.isreal(xq), np.isnan(xq))==False")
            vq = horner_vec_complex(v,xq);
    else:
        raise NotImplementedError("No other method than \'horner\' implemented!")

    if not done:
        raise NotImplementedError("No other method than \'horner\' implemented!")

    if(legacy):
        # vq = reshape(vq,outSz);
        raise NotImplementedError("I think i didn't implement the legacy-mode "
                                  "of mathfun.interpft1().")
    return vq
    """
    case 'horner_freq'
        xq = xq*2;
        if(isreal(xq))
            org_sz = size(xq);
            v = v(:,:);
            xq = xq(:,:);
            vq = horner_vec_real_freq_simpler(v,xq);
            vq = reshape(vq,org_sz);
        else
            vq = horner_vec_complex_freq(v,xq);
        end
    case 'horner_complex'
        xq = xq*2;
        vq = horner_vec_complex(v,xq);
    case 'horner_complex_freq'
        xq = xq*2;
        vq = horner_vec_complex_freq(v,xq);
    case 'mmt'
        % domain [0,2*pi)
        xq = xq*2*pi;
        vq = matrixMultiplicationTransform(v,xq);
    case 'mmt_freq'
        xq = xq*2*pi;
        vq = matrixMultiplicationTransformFreq(v,xq);
    otherwise
        done = false;
    end

    if(~done)
        % Use interp1 methods by expanding grid points using interpft
        fineGridFactor = parseFineGridFactor(fineGridFactor,method);
        vft3 = interpft(v,size(v,1)*fineGridFactor);
        vft3 = [vft3(end-2:end,:); vft3(:,:); vft3(1:4,:)];

        % Map indices from [0 1) to [4 size(v,1)*fineGridFactor+4)
        xq = xq.*(size(v,1)*fineGridFactor);
        xq = xq+4;
        if(legacy || iscolumn(xq))
            vq = interp1(vft3,xq,method);
        else
            % break xq into columns and apply to each corresponding column in v 
            vq = cellfun(@(ii,xq) interp1(vft3(:,ii),xq,method),num2cell(1:numel(xq)/size(xq,1)),num2cell(xq(:,:),1),'UniformOutput',false);
            vq = [vq{:}];
            vSz = size(v);
            vq = reshape(vq,[size(xq,1) vSz(2:end)]);
        end
    end
    
    if(legacy)
        vq = reshape(vq,outSz);
    end
    """





"""
function fineGridFactor = parseFineGridFactor(fineGridFactor,method)
    % Courser methods should use a finer grid if none is specified
    if(isempty(fineGridFactor))
        switch(method)
%             case 'horner'
%                 fineGridFactor = NaN;
%             case 'mmt'
%                 fineGridFactor = NaN;
%             case 'linear'
%                 fineGridFactor = 10;
%             case 'nearest'
%                 fineGridFactor = 10;
%             case 'next'
%                 fineGridFactor = 10;
%             case 'previous'
%                 fineGridFactor = 10;
            case 'pchip'
                fineGridFactor = 6;
            case 'cubic'
                fineGridFactor = 6;
            case 'v5cubic'
                fineGridFactor = 6;
            case 'spline'
                fineGridFactor = 3;
            otherwise
                fineGridFactor = 10;
        end
    end
end

function vq = matrixMultiplicationTransform(v,xq)
    vq = matrixMultiplicationTransformFreq(v_h,xq);
end
function vq = matrixMultiplicationTransformFreq(v_h,xq)
%matrixMultiplicationTransform
%
% Adapted from interpft_extrema
    s = size(v_h);
    scale_factor = s(1);

    % Calculate fft and nyquist frequency
    nyquist = ceil((s(1)+1)/2);

    % If there is an even number of fourier coefficients, split the nyquist frequency
    if(~rem(s(1),2))
        % even number of coefficients
        % split nyquist frequency
        v_h(nyquist,:) = v_h(nyquist,:)/2;
        v_h = v_h([1:nyquist nyquist nyquist+1:end],:);
        v_h = reshape(v_h,[s(1)+1 s(2:end)]);
    end
    % Wave number, unnormalized by number of points
    freq = [0:nyquist-1 -nyquist+1:1:-1]';
    
    % calculate angles multiplied by wave number
    theta = bsxfun(@times,xq,shiftdim(freq,-ndims(xq)));
    % waves
    waves = exp(1i*theta);


    % evaluate each wave by fourier coeffient
    % theta and waves have one more dimension than xq, representing
    % frequency
    ndims_waves = ndims(waves); % ndims(xq) + 1 ?
    % permute v_h such that it is a 1 by (array dim of fourier series) by
    %                               length of fourier series
    dim_permute = [ndims_waves 2:ndims_waves-1 1];
    
    % sum across waves weighted by Fourier coefficients
    % normalize by the the number of Fourier coefficients
    vq = sum(real(bsxfun(@times,waves,permute(v_h,dim_permute))),ndims_waves)/scale_factor;
end
"""

def horner_vec_real(v, xq):
    return horner_vec_real_freq(fftpack.fft(v,axis=0),xq)

def horner_vec_complex(v, xq):
    return horner_vec_complex_freq(fftpack.fft(v),xq)

def horner_vec_real_freq(v_h,xq):
    # print(v_h.shape)
    # plt.imshow(v_h[0].real)
    # plt.show(block=False)
    # % v represents the coefficients of the polynomial
    # %   D x N
    # %   D = degree of the polynomial - 1
    # %   N = number of polynomials
    # % xq represents the query points
    # %   Q x N
    # %   Q = number of query points per polynomial
    # %   N = number of polynomials
    # % vq will be a Q x N matrix of the value of each polynomial
    # %    evaluated at Q query points

    s = v_h.shape
    scale_factor = s[0]
    # % Calculate fft and nyquist frequency
    nyquist = int(np.ceil( (s[0]+1) /2 ))

    # % If there is an even number of fourier coefficients, split the nyquist frequency
    if not np.fmod(s[0], 2):
        raise NotImplementedError(" if not np.fmod(s[0],2): not tested in mathfun.")
        #% even number of coefficients
        #% split nyquist frequency
        v_h[nyquist-1,:] = v_h[nyquist-1,:].real /2

    # % z is Q x N
    z = np.exp(1j*np.pi*xq);
    # % vq starts as 1 x N
    vq = np.expand_dims(v_h[nyquist-1],0)

    #plt.imshow(vq[0].real)
    #plt.show()
    for j in np.arange( nyquist-1, 1, -1):
        vq = z * vq
        vq = v_h[j-1] + vq



    # % Last multiplication
    vq = z * vq # % We only care about the real part
    vq = vq.astype(np.double)

    # % Add Constant Term and Scale
    vq = v_h[0] + vq*2
    vq = vq/scale_factor

    return vq


def horner_vec_complex_freq(v_h, xq):
    raise NotImplementedError("horner_vec_complex_freq is not implemented yet.")

    """
    # % v represents the coefficients of the polynomial
    # %   D x N
    # %   D = degree of the polynomial - 1
    # %   N = number of polynomials
    # % xq represents the query points
    # %   Q x N
    # %   Q = number of query points per polynomial
    # %   N = number of polynomials
    # % vq will be a Q x N matrix of the value of each polynomial
    # %    evaluated at Q query points
    """
    s = v_h.shape
    scale_factor = s[0]

    # % Calculate fft and nyquist frequency
    nyquist = int(np.ceil( (s[0]+1) /2 ))

    # % If there is an even number of fourier coefficients, split the nyquist frequency
    if not np.fmod(s[0],2):
        raise NotImplementedError(" if not np.fmod(s[0],2): not tested in mathfun.")
        # % even number of coefficients
        # % split nyquist frequency
        v_h[nyquist-1,:] = v_h[nyquist-1,:] / 2



    # % z is Q x N
    z = np.exp(1j*np.pi*xq);
    # % vq starts as 1 x N
    print(nyquist)
    vq = v_h[nyquist]
    print(vq.shape)
    for j in [np.arange(nyquist-1,0,-1),np.arange(s[0]-1,nyquist+1,-1)]:
        print(j.shape)
        vq = z * vq
        vq = v_h[j-1] + vq

    print(vq.shape)
    """
    % Last multiplication
    vq = bsxfun(@times,z,vq); % We only care about the real part
    % Add Constant Term and Scale
    vq = bsxfun(@plus,v_h(nyquist+1,v_h_colon{:}),vq);
    vq = vq./scale_factor;
    """
    #return vq