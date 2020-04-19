
import logging
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


from skimage.morphology import *
from skimage.measure import label, regionprops, regionprops_table
from scipy import misc, fftpack


from OrientationSpaceFilter import OrientationSpaceFilter
from ckLogging import notImplemented
from imageSegmentation import *
from mathfun import *
from nonLocalMaximaSuppressionPrecise import *


def steerableAROSD(I, ip):
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
            plt.show(block=False)
        if ip["mask"]:
            meanResponseMask = np.logical_and(meanResponseMask, ip["mask"])
        if ip["diagnosticMode"]:
            plt.figure()
            plt.title("meanResponseMask")
            plt.imshow(meanResponseMask)
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
    #nlms_single_binary = nlms_single > meanResponse;



