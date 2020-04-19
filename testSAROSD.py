
from skimage import io

from steerableAROSD import *
from parameters import DefaultParams
from ckLogging import notImplemented




if __name__ == "__main__":
    if False:
        image = io.imread("CK001HelaOsmo_20_single.tif")
    else:
        image = io.imread("CK001HelaOsmo_20_single_croped.tif")

    image[image<0] = 0
    ip = DefaultParams()
    ip["diagnosticMode"] = False
    steerableAROSD(image, ip)
    print("End of script.")
    plt.show()


