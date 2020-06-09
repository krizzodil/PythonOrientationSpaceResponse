import colorcet

from skimage.measure import regionprops
from scipy.ndimage import correlate

from OrientationSpaceFilter import OrientationSpaceFilter
from common.imageSegmentation import *
from common.mathfun import *
from common.nonLocalMaximaSuppressionPrecise import *
import orientationSpace.diffusion
import OrientationSpace


def steerableAROSD(I, ip):

    returnPackage = {}

    #if ip["diagnosticMode"]:
        #plt.ion()
    if not ip["response"]:
        if not ip["filter"]:
            F = OrientationSpaceFilter.constructByRadialOrder(
                                            1 / 2 / np.pi / ip["sigma"],
                                            1,
                                            ip["order"],
                                            None)
        else:
            F = ip["filter"]

        R = F * I

    else:
        R = ip["response"]
        F = ip["filter"]

    notImplemented("if (ip.Results.useParallelPool): pool = gcp;")

    #% Obtain Fourier transform
    # CK: the important thing is, that it is 1D fft, for some reason
    a_hat = fftpack.fft(R.a.real, axis=2)

    meanResponse = a_hat[:,:,0]/a_hat.shape[2]

    returnPackage["meanResponse"] = meanResponse
    if ip["diagnosticMode"]:
        plt.figure()
        plt.imshow(meanResponse.real)
        plt.title("meanResponse")
        plt.show(block=False)

    #%% Thresholding
    if ip["meanThreshold"] == None:
        if type(ip["meanThresholdMethod"]) == "string":
            if ip["meanThresholdMethod"] == "otsu":
                meanThresholdMethod = thresholdOtsu
            elif ip["meanThresholdMethod"] == "rosin":
                raise NotImplementedError("thresholdRosin is not implemended")
                meanThresholdMethod = thresholdRosin
            else:
                raise NotImplementedError(
                    "threshold string-eval not implemented")
                meanThresholdMethod = eval(ip["meanThresholdMethod"])
        else:
            # % Assume this is a function
            meanThresholdMethod = ip["meanThresholdMethod"]

        meanThreshold = meanThresholdMethod(meanResponse.real)
    else:
        meanThreshold = ip["meanThreshold"]

    if ip["diagnosticMode"]:
        plt.figure()
        plt.hist(meanResponse.real.flatten(), 500)
        plt.title("meanResponse Histogram")
        plt.xlabel("meanResponse")
        plt.ylabel("count")
        plt.axvline(meanThreshold, color="r", linewidth=2)
        plt.show(block=False)

    # %% Masking
    if ip["nlmsMask"] == None:
        meanResponseMask = meanResponse > meanThreshold
        if ip["diagnosticMode"]:
            plt.figure()
            plt.title("meanResponse > meanThreshold")
            plt.imshow(meanResponseMask)
            io.imsave("py02_meanResponseMask.tif", meanResponseMask.astype(np.int8))
            plt.show(block=False)

        if ip["mask"]:
            meanResponseMask = np.logical_and(meanResponseMask, ip["mask"])
        returnPackage["meanResponseMask"] = meanResponseMask
        if ip["diagnosticMode"]:
            plt.figure()
            plt.title("meanResponseMask")
            plt.imshow(meanResponseMask)
            io.imsave("py03_meanResponseMask.tif",
                      meanResponseMask.astype(np.int8))
            plt.show(block=False)

        # % For NLMS(nlmsMask)
        meanResponseMaskDilated = binary_dilation(
                                    meanResponseMask,
                                    selem=disk(ip["maskDilationDiskRadius"])
                                    )
        if ip["maskFillHoles"]:
            raise NotImplementedError("maskFillHoles not implemented")
            #%meanResponseMaskDilated = imfill(meanResponseMaskDilated, 'holes');
        if ip["mask"]:
            meanResponseMaskDilated = np.logical_and(meanResponseMaskDilated,
                                                     ip["mask"])
        returnPackage["meanResponseMaskDilated"] = meanResponseMaskDilated
        if ip["diagnosticMode"]:
            diag_rp = regionprops(label(meanResponseMaskDilated))
            areas = [region.area for region in diag_rp]
            bboxes = [region.bbox for region in diag_rp]
            area = max(areas)
            index = areas.index(area)
            bbox = bboxes[index]
            diag_rp = diag_rp[index]

            plt.figure()
            plt.title("meanResponseMaskDilated")
            plt.imshow(meanResponseMaskDilated)
            # io.imsave("py04_meanResponseMaskDilated.tif",
            #           meanResponseMaskDilated.astype(np.int8))
            rect = plt.Rectangle((bbox[1],bbox[0]),
                                 bbox[3]-bbox[1],
                                 bbox[2]-bbox[0],
                                 color="r", linewidth=1, fill=False)
            plt.gca().add_patch(rect)
            plt.show(block=False)

        nlmsMask = meanResponseMaskDilated
    else:
        # % User defined nlmsMask
        nlmsMask = ip["nlmsMask"]
        diag_rp  = regionprops(nlmsMask)[0]



    # %% Setup orientation analysis problem
    nanTemplate = np.zeros(nlmsMask.shape)
    nanTemplate[:] = np.NaN
    a_hat = np.rollaxis(a_hat, 2, 0)
    a_hat = a_hat[:, nlmsMask]

    # %% Evaluate single orientation, fast easy case
    R_res = R.getResponseAtOrderFT(ip["responseOrder"],2)
    maximum_single_angle = nanTemplate


    maximum_single_angle[nlmsMask] = wraparoundN(-np.angle(a_hat[1,:])/2 ,
                                                  0,
                                                  np.pi)

    nlms_single = nonLocalMaximaSuppressionPrecise(R_res.a.real,
                                                   maximum_single_angle,
                                                   mask = nlmsMask);
    nlms_single_binary = nlms_single > meanResponse

    returnPackage["maximum_single_angle_map"] = maximum_single_angle
    returnPackage["nlms_single"] = nlms_single
    if ip["diagnosticMode"]:
        cm = colorcet.m_CET_C2s   #m_CET_CBC2 matlab original


        maximum_single_angle_map = OrientationSpace.blendOrientationMap(
            maximum_single_angle, R_res.interpft1(maximum_single_angle), cm);

        plt.figure()
        plt.title("maximum_single_angle")
        plt.imshow(maximum_single_angle_map)
        # forSave = maximum_single_angle_map / np.nanmax(maximum_single_angle_map) * 255
        # io.imsave("py05_maximum_single_angle_map.tif",
        #           forSave.astype(np.int8))
        plt.show(block=False)

        plt.figure()
        plt.title("nlms_single")
        plt.imshow(nlms_single)
        io.imsave("py06_nlms_single.tif",
                  nlms_single.astype(np.single))
        plt.show(block=False)

        plt.figure()
        plt.title("nlms_single_binary")
        plt.imshow(nlms_single_binary)
        io.imsave("py07_nlms_single_binary.tif",
                  nlms_single_binary.astype(np.int8))
        plt.show(block=False)

    # %% Determine nlmsThreshold
    if not np.all(ip["nlmsThreshold"]):
        # %% Attenuate meanResponse by neighbor occupancy
        nhood_filter = np.array([[1, 1, 1],[1, 0, 1],[1, 1, 1]])
        nhood_occupancy = correlate(
            nlms_single_binary.astype(np.double),
            nhood_filter,
            mode='constant',
            cval=0
            )/8
        # % double the occupancy for accelerated attenuation
        nhood_occupancy = nhood_occupancy*2
        attenuatedMeanResponse = (1-nhood_occupancy)*meanResponse;
        attenuatedMeanResponse[attenuatedMeanResponse<0] = 0

        nlmsThreshold = attenuatedMeanResponse
    else:
        # % User defined nlmsThreshold
        nlmsThreshold = ip["nlmsThreshold"]

    if ip["diagnosticMode"]:
        plt.figure()
        plt.title("nlmsThreshold")
        plt.imshow(nlmsThreshold.real)
        io.imsave("py08_nlmsThreshold.tif",
                  nlmsThreshold.real.astype(np.single))
        plt.show(block=False)

    # %% Calculate high resolution maxima

    return returnPackage
    """
    # % Adapt length
    if ip["adaptLengthInRegime"]:
        # % Find orientation maxima with nlmsMask only
        interpftValues = interpft_extrema(a_hat,
                                          0,
                                          True,
                                          None,
                                          False
                                          );
        maxima_highest_temp = interpftValues["maxima"]
        minima_highest_temp = interpftValues["minima"]

        # plt.figure()
        # # plt.imshow(maxima_highest_temp[:,0:100])
        # plt.hist(maxima_highest_temp.flatten(), range=(-500,500))
        # plt.show(block=False)
        # plt.figure()
        # # plt.imshow(minima_highest_temp[:,0:100])
        # plt.hist(minima_highest_temp.flatten(), range=(-500,500))
        # plt.show()

        # % Count
        n_maxima_highest_temp = maxima_highest_temp.shape[0] \
                                - np.sum(np.isnan(maxima_highest_temp), 0)
        K_high = F.K
        K_low = np.maximum(n_maxima_highest_temp - 1,
                           ip["responseOrder"])
        #warning('off','halleyft:maxIter');
        K_high, K_low = orientationSpace.diffusion.findRegimeBifurcation(
                                                        a_hat,
                                                        F.K,
                                                        K_high,
                                                        K_low,
                                                        maxima_highest_temp,
                                                        minima_highest_temp,
                                                        None,
                                                        0.1,
                                                        True)

        
        best_derivs = orientationSpace.diffusion.orientationMaximaFirstDerivative(a_hat,F.K,maxima_highest_temp);
        best_abs_derivs = abs(best_derivs);
        best_K = repmat(F.K,size(best_derivs));
        best_maxima = maxima_highest_temp;
        maxima_working = maxima_highest_temp;
        for K=F.K:-ip.Results.K_sampling_delta:1
            s = K > K_high;
            lower_a_hat = orientationSpace.getResponseAtOrderVecHat(a_hat(:,s),F.K,K);
            [new_derivs(:,s),~,maxima_working(:,s)] = orientationSpace.diffusion.orientationMaximaFirstDerivative(lower_a_hat,K,maxima_working(:,s),[],true);
            new_abs_derivs(:,s) = abs(new_derivs(:,s));
            better(:,s) = new_abs_derivs(:,s) < best_abs_derivs(:,s);

            % Update better
            best_abs_derivs(better) = new_abs_derivs(better);
            best_derivs(better) = new_derivs(better);
            best_K(better) = K;
            best_maxima(better) = maxima_working(better);
        end

        maxima_highest_temp = best_maxima / 2;
    else
        % Find orientation maxima with nlmsMask only
        maxima_highest_temp = interpft_extrema(a_hat,1,true,[],false)/2;
        best_K = repmat(F.K,size(maxima_highest_temp));
    end

    maxima_highest = nanTemplate(:,:,ones(size(maxima_highest_temp,1),1));
    maxima_highest = shiftdim(maxima_highest,2);
    for i=1:size(maxima_highest_temp,1)
        maxima_highest(i,nlmsMask) = maxima_highest_temp(i,:);
    end
    maxima_highest = shiftdim(maxima_highest,1);
        """
