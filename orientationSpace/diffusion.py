import numpy as np
from scipy import fftpack
from common.mathfun import *

# function [K_high,K_low,extrema_high,extrema_low] = findRegimeBifurcation(response,K_response,K_high,K_low,maxima,minima,maxIterations,tolerance,freq,debugIdx)

# % REMARKS
# % This function uses halleyft to quickly find extrema at the midpoint
# % between K_high and K_low using the extrema at K_midpoint

def findRegimeBifurcation(response,
                          K_response,
                          K_high,
                          K_low = np.array([]),
                          maxima = np.array([]),
                          minima = np.array([]),
                          maxIterations = 10,
                          tolerance = None,
                          freq = False,
                          debugIdx = np.array([])):

    if np.size(K_response) == 1:
        K_response = np.tile(K_response, (1,response.shape[1]))

    if K_low.size == 0:
        K_low = np.ones( (1, response.shape[1]) )


    if not np.any(tolerance):
        tolerance = np.maximum( np.spacing(K_high),
                                (K_high-K_low) / 2**maxIterations )
    if debugIdx.size == 0:
        debug = False
    else:
        debug = True

    if freq:
        response_hat = response
    else:
        response_hat = fftpack.fft(response)

    if maxima.size == 0:
        # % Calculate maxima and minima from response_hat if not given
        interprftValues = interpft_extrema(response_hat, 1, False, None, False)
        maxima = interprftValues["maxima"]
        minima = interprftValues["minima"]

    # % Use minima allows for some error checking since maxima and minima
    # % should occur in pairs
    useMinima = False if minima.size == 0 else True

    # % Make K_high match 2nd dimension of response
    if np.size(K_high) == 1:
        K_high = np.tile(K_high, (1,response.shape[1]))

    # % Make K_low match 2nd dimension of response
    if np.size(K_low) == 1:
        K_low = np.tile(K_low, (1,response.shape[1]))

    if useMinima:
        # % Combine maxima and minima into single array
        extrema_working = np.sort( np.concatenate( (maxima, minima), axis=0 ),
                                   axis=0 )[::-1,:]
        plt.figure()
        plt.title("extrema_working")
        plt.hist(extrema_working.real.flatten(), bins=500, range=(-1000, 1000))
        plt.show()
        # % Pad extrema so that it matches first dimension of response
        #extrema_working(end+1:size(response,1),:) = NaN;
    else:
        extrema_working = maxima
    """
    # % Initialize output to input, should be the same size
    extrema_high = extrema_working;

    nExtrema_working = size(extrema_working,1) - sum(isnan(extrema_working));
    % If there are one or less extrema we are done
    not_done = nExtrema_working > 1;

    % The working variables represent the data we are not done with
    K_high_working = K_high(:,not_done);
    K_low_working = K_low(:,not_done);
    K_response_working = K_response(:,not_done);
    extrema_working = extrema_working(:,not_done);
    response_working_hat = response_hat(:,not_done);

    for i=1:maxIterations
        % Delegate the real work to a stripped down function
        [K_high_working,K_low_working,extrema_working] = ...
            findRegimeBifurcationHat( ...
                response_working_hat,K_response_working, ...
                K_high_working,K_low_working, ...
                extrema_working,useMinima);

        % Update the output variables with the data we are not done with
        K_high(not_done) = K_high_working;
        K_low(not_done) = K_low_working;
        extrema_high(:,not_done) = extrema_working;

        % We are not done if the difference in K exceeds the tolerance
        not_done_working = K_high_working - K_low_working > tolerance;
        not_done(not_done) = not_done_working;

        % Update the working variables
        K_response_working = K_response_working(not_done_working);
        K_high_working = K_high_working(not_done_working);
        K_low_working = K_low_working(not_done_working);
        response_working_hat = response_working_hat(:,not_done_working);
        extrema_working = extrema_working(:,not_done_working);

        if(debug)
            K_high(debugIdx)
            K_low(debugIdx)
            extrema_high(:,debugIdx)
        end

        if(isempty(K_high_working))
            if(debug)
                i
            end
            break;
        elseif(debug)
            length(K_high_working)
        end


    end


    if(nargout > 3)
        response_low_hat = orientationSpace.getResponseAtOrderVecHat(response_hat,K_response,K_low);
        [maxima_low,minima_low] =  interpft_extrema(response_low_hat,1,false,[],false);
        extrema_low = sort([maxima_low,minima_low]);
    end
end

function [K_high,K_low,extrema_high] = findRegimeBifurcationHat(response_hat,K_response,K_high,K_low,extrema,useMinima)
    nExtrema = size(extrema,1) - sum(isnan(extrema));
    K_midpoint = (K_high+K_low)/2;
    response_midpoint_hat = orientationSpace.getResponseAtOrderVecHat(response_hat,K_response,K_midpoint);

    [extrema_midpoint,xdg] = halleyft(response_midpoint_hat,extrema,true,1,1e-12,10,true,1e-4);

    % Only keep maxima
    if(~useMinima)
        extrema_midpoint(xdg(:,:,2) > 0) = NaN;
    end

    % Should be done by halleyft with uniq = true
    % Eliminate duplicates
    extrema_midpoint = sort(extrema_midpoint);
    max_extrema_midpoint = max(extrema_midpoint)-2*pi;
    extrema_midpoint(diff([max_extrema_midpoint; extrema_midpoint]) < 1e-4) = NaN;
    extrema_midpoint = sort(extrema_midpoint);

    nExtremaMidpoint = size(extrema_midpoint,1) - sum(isnan(extrema_midpoint));

    % Do error correction
    if(useMinima)
        % Maxima and minima should occur in pairs.
        % An odd number of extrema would indicate an error
        oddIdx = mod(nExtremaMidpoint,2) == 1;
        % Find extrema that are close together, which may indicate an error
        closeExtremaIdx = any(diff([max(extrema_midpoint)-2*pi; extrema_midpoint]) < 0.02);

        oddIdx(closeExtremaIdx) = true;
        if(any(oddIdx))
            [odd_maxima,odd_minima] = interpft_extrema(response_midpoint_hat(:,oddIdx),1,true,[],false);
            oddExtrema = [odd_maxima; odd_minima];
            oddExtrema = sort(oddExtrema);
            oddExtrema = oddExtrema(1:min(size(extrema_midpoint,1),end),:);
            sExtrema = size(oddExtrema,1);
            extrema_midpoint(1:sExtrema,oddIdx) = oddExtrema;
            extrema_midpoint(sExtrema+1:end,oddIdx) = NaN;
            nExtremaMidpoint(oddIdx) = size(extrema_midpoint,1) - sum(isnan(extrema_midpoint(:,oddIdx)));
        end
    end

    if(useMinima)
        bifurcationInHigh = nExtrema - nExtremaMidpoint >= 2;
    else
        bifurcationInHigh = nExtremaMidpoint ~= nExtrema;
    end
    bifurcationInLow = ~bifurcationInHigh;
    K_low(bifurcationInHigh) = K_midpoint(bifurcationInHigh);
    K_high(bifurcationInLow) = K_midpoint(bifurcationInLow);
    extrema_high = extrema;
    extrema_high(:,bifurcationInLow) = sort(extrema_midpoint(:,bifurcationInLow));
end



"""