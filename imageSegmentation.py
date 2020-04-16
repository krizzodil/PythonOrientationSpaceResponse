
import numpy as np
from ckLogging import *
from skimage.filters import threshold_otsu

def thresholdOtsu(imageIn):
    # the implementation in the original code seems
    # unnecessarily complicated
    notImplemented("custom otsu not implemented; using skimage.filters.")
    return threshold_otsu(imageIn)
"""


%get minumum and maximum pixel values in image
minSignal = min(imageIn(nzInd));
maxSignal = max(imageIn(nzInd));

%normalize nonzero value between 0 and 1
imageInNorm = zeros(size(imageIn));
imageInNorm(nzInd) = (imageIn(nzInd)- minSignal) / (maxSignal - minSignal);

level = graythresh(imageInNorm);

level = level*(maxSignal - minSignal)+minSignal;

if showPlots
    imageMask = imageIn >= level;
    figure;
    imagesc(imageIn);
    hold on
    contour(imageMask,'w')
    colormap hot
end

if nargout > 1
    varargout{1} = double(imageIn >= level);
end

"""

