import logging
import os
from skimage import io

from steerableAROSD import *
from parameters import DefaultParams
from ckLogging import notImplemented




if __name__ == "__main__":
    logging.root.setLevel(logging.INFO)
    images = []
    if False:
        images.append( io.imread("CK001HelaOsmo_20_single.tif"))
    elif False:
        images.append( io.imread("CK001HelaOsmo_20_single_croped.tif"))
    else:
        original = io.imread("CK001HelaOsmo_20_cropped_comp_recon.tif")
        rcOriginal = original[:, 0, :, :]
        for i in range(rcOriginal.shape[0]):
            images.append(rcOriginal[i, :, :])

    root = os.getcwd()
    for i, image in enumerate(images):
        newPath =  os.path.join(root, f"image_{i}")
        try:
            os.mkdir( newPath )
        except:
            pass
        os.chdir(newPath)
        image[image<0] = 0
        ip = DefaultParams()
        #ip["diagnosticMode"] = False
        steerableAROSD(image, ip)

    print("End of script.")
    plt.show()


