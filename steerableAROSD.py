
import logging
import copy

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import morphology
from scipy import misc, fftpack

from parameters import DefaultParams
from OrientationSpaceFilter import OrientationSpaceFilter
from ckLogging import notImplemented
from imageSegmentation import *



def tprint(string):
    if TEST:
        print(string)

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
        meanResponseMaskDilated = morphology.binary_dilation(
                        meanResponseMask,
                        selem=morphology.disk(ip["maskDilationDiskRadius"])
                        )




if __name__ == "__main__":
    image = io.imread("CK001HelaOsmo_20_single_croped.tif")
    image[image<0] = 0
    ip = DefaultParams()
    steerableAROSD(image, ip)
    print("End of script.")
    plt.show()
