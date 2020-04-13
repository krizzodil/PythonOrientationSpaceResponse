import logging

import numpy as np
import matplotlib.pyplot as plt

from scipy import misc, fftpack

from ckLogging import notImplemented
import OrientationSpace
import OrientationSpaceResponse


class OrientationSpaceFilter:
    def __init__(self, f_c, b_f, K, normEnergy=None):
        # the original from Mark can take arrays for the values below, but my
        # port cannot. Should it be necessary, I will incorporate the required
        # functionality in another class managing an array containing instances
        # of this class.

        self.f_c = f_c                  # radial central frequency
        self.b_f = b_f                  # radial frequency bandwidth
        self.K   = K                    # angular order
        self.normEnergy = normEnergy    # normilization setting
        self.sampleFactor = 1           # multiplier to calculate n from K
        self.size = None                # filter size, should correspond with image size
        self.F = None                   # Filter itself
        self.isSetup = False      #My addition to the code

        if b_f == None:
            self.b_f = 1/np.sqrt(2) * f_c

        self.n = 2 * self.sampleFactor * np.ceil(self.K) + 1;

        """
        self.n                  # number of angular filter templates    
        self.angles             # basis angles
        self.F                  # Filter itself
        self.angularGaussians   # Angular gaussians useful for manipulating response
        self.centralFrequency   # f_c
        self.frequencyBandwidth # b_f
        self.order              # K
        """

    @property
    def angles(self):
        return np.arange(self.n)/self.n*np.pi

    @property
    def centralFrequency(self):
        return self.f_c

    @property
    def frequencyBandwidth(self):
        return self.b_f

    @property
    def order(self):
        return self.K


    def __mul__(self, other):
        """ Overriding *-operator
        :param other: np.ndarray image
        :return: R
        """
        logging.warning("Overridded *-operator!")

        # Convolution
        R = self.getResponse(other)
        return R

    def getResponse(self, image):
        If = fftpack.fft2(image)
        ridgeResponse   = self.applyRidgeFilter(If)
        edgeResponse    = self.applyEdgeFilter(If)
        angularResponse = ridgeResponse + edgeResponse
        R = OrientationSpaceResponse.OrientationSpaceResponse(self, angularResponse)
        return R

    def getEnergy(self):
        raise notImplemented("getEnergy not implemented!")
        self.requireSetup()
        s = self.F.shape
        notImplemented("s(end+1:3) = 1;")
        F = np.reshape(self.F, (s[0]*s[1], s[2]) )
        E = np.sqrt( np.sum(F.real**2, 0) ) \
            + 1j * np.sqrt( np.sum(F.imag**2, 0) )

        print(E)
        #E = E. / sqrt(s(1) * s(2));
    """        function E = getEnergy(obj)

            requireSetup(obj);
            s = size(obj.F);
            s(end+1:3) = 1;
            F = reshape(obj.F,s(1)*s(2),s(3));
            E = sqrt(sum(real(F).^2)) + 1j*sqrt(sum(imag(F).^2));
            E = E ./ sqrt(s(1)*s(2));"""
    #return E


    def getAngularKernel(self, coords=None):
        if coords == 0:
            coords = OrientationSpace.getFrequencySpaceCoordinates(self.size)
        A = OrientationSpace.angularKernel(self.K, self.angles, coords);
        return A

    def getRadialKernel(self, coords):
        if coords == 0:
            coords = OrientationSpace.getFrequencySpaceCoordinates(self.size)
        R = OrientationSpace.radialKernel(self.f_c, self.b_f, coords);
        return R



    def setupFilter(self, siz):
        if self.isSetup:
            logging.info("Filter is already set up.")
            return 0
        coords = OrientationSpace.getFrequencySpaceCoordinates(siz)

        A = self.getAngularKernel(coords)
        R = self.getRadialKernel(coords)
        self.F = A * R[:,:,None]

        notImplemented("There might be a problem with the "
                         "original duality of obj.F and obj(i).F")

        if self.normEnergy:
            raise NotImplementedError("Sorry, I didn't implement the normEnergy!")
            if   self.normEnergy == "energy":
                self.getEnergy()

            elif self.normEnergy == "peak":
                pass
            elif self.normEnergy == "scale":
                pass
            elif self.normEnergy == "sqrtscale":
                pass
            elif self.normEnergy == "n":
                pass
            elif self.normEnergy == "none":
                pass
            elif type(self.normEnergy) == "string":
                raise ValueError("Invalid normEnergy property")

        self.isSetup = True
    """

            switch(obj(o).normEnergy)
                case 'energy'
                    % E is complex
                    E = shiftdim(obj(o).getEnergy(),-1);
                    F = obj(o).F;
                    obj(o).F = bsxfun(@rdivide,real(F),real(E)) +1j*bsxfun(@rdivide,imag(F),imag(E));
                case 'peak'
                    F = obj(o).F;
                    sumF = sum(F(:))./numel(F);
                    obj(o).F = real(F)./real(sumF) + 1j*imag(F)./imag(sumF);
                case 'scale'
                    obj(o).F = obj(o).F ./ obj(o).f_c ./ sqrt(siz(1)*siz(2));
                case 'sqrtscale'
                    obj(o).F = obj(o).F ./ sqrt(obj(o).f_c) ./ sqrt(siz(1)*siz(2));
                case 'n'
                    obj(o).F = obj(o).F ./ obj(o).n;
                case 'none'
                otherwise
                    error('OrientationSpaceFilter:setupFilterNormEnergy', ...
                        'Invalid normEnergy property');
            end
        end
    end"""



    def applyRidgeFilter(self, If):
        self.setupFilter(If.shape)
        #ridgeResponse = np.fft.fft2( If[:,:, None]*self.F.real,
        #                             axes=(0,1)).real
        ridgeResponse = fftpack.ifft2( If[:,:, None]*self.F.real,
                                     axes=(0,1)).real
        return ridgeResponse


    def applyEdgeFilter(self, If):
        self.setupFilter(If.shape)
        #edgeResponse = 1j*( np.fft.ifft2( (If*-1j)[:,:,None]*self.F.imag,
        #                                   axes=(0,1))).real
        edgeResponse = 1j*(( fftpack.ifft2( (If*-1j)[:,:,None]*(self.F.imag),
                                           axes=(0,1))).real)
        return edgeResponse

    def requireSetup(self):
        if self.F.size == 0:
            logging.error('OrientationSpaceFilter:NotSetup',
                          'Filter must be setup in order for this operation to succeed.')
            raise ValueError

    def constructByRadialOrder(f_c, K_f, K, normEnergy=None,
                               constructor=None #not implemented
                               ):
        b_f = f_c / np.sqrt(K_f)
        if constructor == None:
            F = OrientationSpaceFilter(f_c, b_f, K, normEnergy)
        else:
            raise NotImplementedError
            F = constructor(f_c, b_f, K, normEnergy)

        #notImplemented("F = F(logical(eye(length(f_c))))")
        return(F)

