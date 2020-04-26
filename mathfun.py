import numpy as np
import logging
#from multiprocessing import Pool
import concurrent.futures
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

    elif method=="horner_freq":
        xq = xq * 2;
        if np.all(np.logical_or(np.isreal(xq), np.isnan(xq))):
            logging.info("Mathfun.interpft1() "
                         "np.all(np.logical_or(np.isreal(xq), np.isnan(xq))==True")
            org_sz = xq.shape
            #v = v[]:,:);
            #xq = xq(:,:);
            vq = horner_vec_real_freq_simpler(v, xq);
            #vq = reshape(vq, org_sz);
        else:
            logging.info("Mathfun.interpft1() "
                         "np.all(np.logical_or(np.isreal(xq), np.isnan(xq))==False")
            vq = horner_vec_complex_freq(v, xq);

    else:
        raise NotImplementedError("Method not implemented!")

    if not done:
        raise NotImplementedError("Method not implemented!")

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
    vq = v_h[nyquist]

    for j in [np.arange(nyquist-1,0,-1),np.arange(s[0]-1,nyquist+1,-1)]:
        vq = z * vq
        vq = v_h[j-1] + vq

    """
    % Last multiplication
    vq = bsxfun(@times,z,vq); % We only care about the real part
    % Add Constant Term and Scale
    vq = bsxfun(@plus,v_h(nyquist+1,v_h_colon{:}),vq);
    vq = vq./scale_factor;
    """
    #return vq


def get_roots(arr):
    #r = np.zeros( arr.shape, dtype= np.complex128 )
    # arr.shape[0]-1 corresponds to matlab original output_size1
    r = np.zeros( (arr.shape[0]-1, arr.shape[1]), dtype= np.complex128 )
    for j in range(arr.shape[1]):
        try:
            roots =  np.roots( arr[:, j] )
            r[:, j] = roots
        except Exception as err:
            print(err)
            r[:,j] = np.NaN
    return r

def sortMatrices(template, matrices):
    sortIndices = template.argsort(axis=0)
    outMatrices = []
    for matrix in matrices:
        matrix = matrix[np.arange(matrix.shape[0])[:, np.newaxis],
                        sortIndices[::-1, :]]
        outMatrices.append(matrix)
    return outMatrices


def interpft_extrema(x,dim,sorted=False,TOL=None,dofft=True):
    values = {"maxima":         0,
              "minima":         0,
              "maxima_value":   0,
              "minima_value":   0,
              "other":          0,
              "other_value:":   0
              }

    if TOL == None:
        TOL = -np.spacing(np.array([1], np.single)) *1000 # x.real.dtype))
        #  -1.1921e-04 in matlab single (but is complex

    if x.ndim > 2:
        raise ValueError("interpft_extrema: x has too many dimensions, max 2.")
    if isvector(x) and x.shape[0] > x.shape[1]:
        raise NotImplementedError("interpft_extrema:"
                                  "if isvector(x) and x.shape[0] "
                                  "> x.shape[1]: not tested.")
        unshift = 1 # NEEDS MODIFICATION IN PYTHON
        x.dot(x.transpose())
    else:
        unshift = 0 # DON'T MOVE AXES

    if dim != 0:
        raise NotImplementedError("interpft_extrema: dim != 1 not implemented.")
        """
        D do we need that?
        if(nargin > 1 && ~isempty(dim))
            x = shiftdim(x,dim-1);
            unshift = ndims(x) - dim + 1;
        """

    output_size = list(x.shape)
    output_size[0] = output_size[0] - 1
    s = x.shape
    scale_factor = s[0]
    
    if s[0]== 1:
        maxima = np.zeros(s)
        minima = maxima[:]
        other = maxima[:]
        maxima_value = x[:]
        minima_value = maxima_value[:]
        other_value = maxima_value[:]
        values = {"maxima":         maxima,
                  "minima":         minima,
                  "maxima_value":   other,
                  "minima_value":   maxima_value,
                  "other":          minima_value,
                  "other_value:":   other_value
                  }
        return values

    # % Calculate fft and nyquist frequency
    if dofft:
        x_h = fftpack.fft(x)
    else:
        x_h = x

    nyquist = int(np.ceil((s[0]+1)/2))

    #% If there is an even number of fourier coefficients,
    #  split the nyquist frequency
    if not np.fmod(s[0],2):
        raise NotImplementedError("interpft_extrema: if not fmod(s[0],2):")
        # % even number of coefficients
        # % split nyquist
        x_h[nyquist-1,:] = x_h[nyquist-1,:]/2
        """
        x_h = x_h([1:nyquist nyquist nyquist+1:end] , :  )
        x_h = reshape(x_h,[s(1)+1 s(2:end)]);
        output_size(1) = output_size(1) + 1;
        """
    # % Wave number, unnormalized by number of points
    freq = np.array( list(range(0, nyquist)) + list(range(-nyquist+1, 0)) )[:,np.newaxis]

    # % calculate derivatives, unscaled by 2*pi
    # since we are only concerned with the signs of the derivatives
    dx_h  = x_h * ( freq * 1j)
    dx2_h = x_h * (-freq ** 2)

    # % use companion matrix approach
    dx_h = - fftpack.fftshift(dx_h,0)
    # dx_h = dx_h(:,:) # useless line?

    output_size1 = output_size[0]
    nProblems = np.prod(output_size[1:])
    batchSize = min([1024, nProblems])
    nBatches = int(np.ceil(nProblems/batchSize))

    # % Only use parallel workers if a pool already exists
    # nWorkers = ~isempty(gcp('nocreate'))*nBatches;
    #inarr = np.ones((1,nBatches))  * batchSize
    #inarr[-1] = inarr[-1] + nProblems - np.sum(inarr)

    starts = list(range(0, nProblems-batchSize, batchSize))
    ends = starts[1:] + [dx_h.shape[1]]
    dx_h = [dx_h[:,start:end] for start, end in zip(starts, ends)]

    #r = [False] * len(dx_h)
    logging.info("interpft_extrema: Starting block with np.roots. ")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(get_roots, dx_h)
        logging.warning("Dimensions of last results-element "
                        "might need modification!")
    logging.info("interpft_extrema: block with np.roots finished. ")

    r = np.concatenate(list(results), axis=1)

    # % magnitude
    magnitude = np.absolute(np.log( np.absolute(r) ))
    # % keep only the real answers
    real_map = magnitude <= np.absolute(TOL)
    # % If tolerance is negative and no roots are found, then use the
    # % root that is closest to being real
    if TOL < 0:
        no_roots = np.invert( np.any(real_map, axis=0) )

        real_map[:,no_roots] = np.less_equal(magnitude[:,no_roots],
                                             np.min(magnitude[:,no_roots],
                                                    axis=0) *10
                                             )
    # % In the call to roots the coefficients were entered in reverse order (negative to positive)
    # % rather than positive to negative. Therefore, take the negative of the angle..
    # % angle will return angle between -pi and pi
    
    r = -np.angle( r[real_map] )
    
    # % Map angles to between 0 and 2 pi, moving negative values up
    # % a period
    neg_extrema = r < 0
    r[neg_extrema] = r[neg_extrema] + 2*np.pi

    extrema = np.empty(output_size)
    extrema[:] = np.NaN
    extrema[real_map] = r

    # % Use Horner's method to compute 2nd derivative value
    dx2 = interpft1(np.array([0, 2*np.pi]),dx2_h,extrema,'horner_freq');

    # % dx2 could be equal to 0, meaning inconclusive type
    # % local maxima is when dx == 0, dx2 < 0
    output_template = np.empty( output_size )
    output_template[:] = np.NaN

    maxima = output_template[:]
    minima = output_template[:]
    
    maxima_map = dx2 < 0
    minima_map = dx2 > 0
    
    maxima[maxima_map] = extrema[maxima_map]
    # % local minima is when dx == 0, dx2 > 0
    minima[minima_map] = extrema[minima_map]

    if sorted:
	    # % calculate the value of the extrema if needed
        
        # % Use Horner's method to compute value at extrema
        extrema_value = interpft1(np.array([0, 2*np.pi]),
                                  x_h,
                                  extrema,
                                  'horner_freq')
        # % interpft1 will automatically scale, rescale if needed
        if scale_factor != x_h.shape[0]:
            extrema_value = extrema_value/scale_factor*x_h.shape[0]

        maxima_value = output_template[:]
        maxima_value[maxima_map] = extrema_value[maxima_map]

        minima_value = output_template[:]
        minima_value[minima_map] = extrema_value[minima_map]

        maxima_value_inf = maxima_value[:]
        maxima_value_inf[np.isnan(maxima_value_inf)] = -np.inf

        maxima, maxima_value = sortMatrices(maxima_value_inf,
                                            [maxima, maxima_value])
        maxima_value_inf = 0 # clear
        numMax = np.nansum(maxima_map, axis=0)
        numMax = np.nanmax(numMax)
        maxima = maxima[0:numMax,:]
        maxima_value = maxima_value[0:numMax,:]

        minima_value_inf = minima_value[:]
        minima_value_inf[np.isnan(minima_value_inf)] = np.inf

        minima, minima_value = sortMatrices(minima_value_inf,
                                            [minima, minima_value])

        
        minima_value_inf = 0 # clear
        numMin = np.nansum(minima_map, axis=0)
        numMin = np.nanmax(numMin)
        minima = minima[0:numMin,:]
        minima_value = minima_value[0:numMin,:]

        #maxima_value = shiftdim(maxima_value,unshift);
        #minima_value = shiftdim(minima_value,unshift);

        """
        if(nargout > 4)
            % calculate roots which are not maxima or minima
            other = output_template;
            other_map = dx2 == 0;
            other(other_map) = extrema(other_map);
            
            other_value = output_template;
            other_value(other_map) = extrema_value(other_map);
            
            if(sorted)
                other_value_inf = other_value;
                other_value_inf(isnan(other_value_inf)) = -Inf;
                [~,other,other_value] = sortMatrices(other_value_inf,other,other_value,'descend');
                clear other_value_inf;
            end
            
            #other = shiftdim(other,unshift);
            #other_value = shiftdim(other_value,unshift);
        end
    end
    """


    """
    NOT IMPLEMENTED
    if(nargout == 0)
        % plot if no outputs requested
        if(sorted)
            real_maxima = maxima(:);
            real_maxima_value = maxima_value(:);
        else
            real_maxima = maxima(maxima_map);
            real_maxima_value = real(maxima_value(maxima_map));
        end
        if(~isempty(real_maxima))
        % Maxima will be green
            plot([real_maxima real_maxima]',ylim,'g');
            plot(real_maxima,real_maxima_value,'go');
        end
        if(sorted)
            real_minima = minima(:);
            real_minima_value = minima_value(:);
        else
            real_minima = minima(minima_map);
            real_minima_value = real(minima_value(minima_map));
        end
        if(~isempty(real_minima))
	    % Minima will be red
            plot([real_minima real_minima]',ylim,'r');
            plot(real_minima,real_minima_value,'ro');
        elseif(~isempty(real_maxima))
            warning('No extrema');
        end
    end
    """

    #maxima = shiftdim(maxima,unshift)
    #minima = shiftdim(minima,unshift)

    values = {"maxima": maxima,
              "minima": minima,
              #"maxima_value": other,
              #"minima_value": maxima_value,
              #"other": minima_value,
              #"other_value:": other_value
              }
    return values



def horner_vec_real_freq_simpler(v_h, xq):
    nQ = xq.shape[0]
    s = v_h.shape
    scale_factor = s[0]

    # % Calculate fft and nyquist frequency
    nyquist = int(np.ceil((s[0]+1)/2))

    # % If there is an even number of fourier coefficients,
    # split the nyquist frequency
    if not np.fmod(s[0],2):
        raise NotImplementedError("interpft_extrema: if not fmod(s[0],2):")
        # % even number of coefficients
        # % split nyquist frequency
        v_h[nyquist-1,:] = x_h[nyquist-1,:].real/2

    # % z is Q x N
    z = np.exp(1j*np.pi*xq)
    # % vq starts as 1 x N
       
    vq = v_h[nyquist-1,:]
    vq = np.tile(vq, (nQ, 1)) #repmat(vq,nQ,1);
    for j in range(nyquist-1, 1, -1):
        vq = z*vq
        vq = np.tile(v_h[j,:], (nQ, 1)) + vq #repmat(v_h(j,:),nQ,1)+vq;
       
    # % Last multiplication
    vq = z * vq
    vq = vq.real
    # % Add Constant Term and Scale
    vq = np.tile(v_h[0,:], (nQ, 1)) + vq*2 #repmat(v_h(1,:),nQ,1)+vq*2;
    vq = vq / scale_factor
    return vq
