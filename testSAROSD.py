import logging
import os
import glob
import itertools
import concurrent

import numpy as np
from skimage import io, img_as_float32
import napari


from steerableAROSD import steerableAROSD
from parameters import DefaultParams
from ckLogging import notImplemented

def show(image, *args, **kwargs):
    with napari.gui_qt():
        image[np.isnan(image)] = 0
        # create a Viewer and add an image here
        viewer = napari.view_image(image, *args, **kwargs)

def getIOname(oriPath, modification):
    root, filename = os.path.split(oriPath)
    filename, ext  = os.path.splitext(filename)
    return os.path.join(root, modification, filename + "_" + modification + ext)


if __name__ == "__main__":
    logging.root.setLevel(logging.INFO)

    ip = DefaultParams()

    os.chdir("D:/InputOutput/AROSDandStretch")
    toLoad = glob.glob("*.tif")
    toLoad = glob.glob("CK001HelaOsmo_20_cropped_comp_recon.tif")
    #toLoad = glob.glob("*croped.tif")

    for loadname in toLoad:
        print(f"Analysing {loadname}")
        original = io.imread(getIOname(loadname, "reg"), plugin='tifffile')

        if original.ndim == 4:
            rcOriginal = original[:, 0, :, :].astype(np.single)
        elif original.ndim == 3:
            if original.shape[0] > original.shape[-1]:
                rcOriginal = np.moveaxis(original, 2, 0).astype(np.single)
            else:
                rcOriginal = original[:, :, :].astype(np.single)

        elif original.ndim == 2:
            rcOriginal = original[np.newaxis,:,:].astype(np.single)

        else:
            print(loadname, original.shape)
            raise ValueError

        print(rcOriginal.shape)
        rcOriginal[rcOriginal < 0] = 0


        images = [rcOriginal[i, :, :] for i in range(rcOriginal.shape[0])]

        if True:
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                packages = executor.map(steerableAROSD,
                                        images,
                                        itertools.repeat(ip, len(images))
                                        )
        else:
            packages = []
            for image in images:
                package = steerableAROSD(image, ip)
                packages.append(package)

        packages = list(packages)
        concats = {}
        for key in packages[0].keys():
            concats[key] = np.array( [package[key] for package in packages] )

        for title, image in concats.items():
            if True:
                if not os.path.exists(title):
                    os.mkdir(title)
                filename = getIOname(loadname, title)
                io.imsave(filename, img_as_float32(image.real), photometric="minisblack")

            if False:
                show(image.real)

    print("End of script.")


