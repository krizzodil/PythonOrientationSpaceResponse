import logging
import copy

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import misc, fftpack

from parameters import DefaultParams
from OrientationSpaceFilter import OrientationSpaceFilter
from ckLogging import notImplemented
from common.imageSegmentation import thresholdOtsu


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
        plt.imshow(meanResponse.real)
        plt.title("meanResponse")
        plt.show(block=False)

    #%% Thresholding
    if ip["meanThreshold"] == None:
        if type(ip["meanThresholdMethod"]) == "string":
            if ip["meanThresholdMethod"] == "otsu":
                meanThresholdMethod = thresholdOtsu
            elif ip["meanThresholdMethod"] == "rosin":
                meanThresholdMethod = thresholdRosin
            else:
                meanThresholdMethod = eval(ip["meanThresholdMethod"])
        else:
            # % Assume this is a function
            meanThresholdMethod = ip["meanThresholdMethod"]

        meanThreshold = meanThresholdMethod(meanResponse)
    else:
        meanThreshold = ip["meanThreshold"]




if __name__ == "__main__":
    image = io.imread("CK001HelaOsmo_20_single_croped.tif")
    ip = DefaultParams()
    steerableAROSD(image, ip)
    print("End of script.")
    plt.show()
