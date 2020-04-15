
import numpy as np
from scipy import fftpack

import OrientationSpaceFilter



class OrientationSpaceResponse:
    def __init__(self, filter, angularResponse):
        self.filter = filter
        self.angularResponse = angularResponse
        self.n = self.angularResponse.shape[2]

    def get_a(self):
        return self.angularResponse
    def set_a(self, a):
        self.angularResponse = a
    a = property(get_a, set_a)


    def getResponseAtOrderFT(self, K_new, normalize=0):
        if K_new == self.filter.K:
            #% copy the response object
            # % For consistency at the moment, return only the real
            # % (ridge) component 2017/11/28
            Response = OrientationSpaceResponse(self.filter,
                                                self.angularResponse.real)
            return Response

        else:
            # % Just for Gaussian calculation;
            n_new = 1 + 2*K_new
            n_old = 1 + 2*self.filter.K

            # % Shouldn't be based off K and not n
            s_inv = np.sqrt( n_old**2 * n_new**2 / (n_old**2 - n_new**2))
            s_hat = s_inv / (2*np.pi)

            if normalize == 2:
                # ck: should the range start at 0 because Python indexing?
                # original: (1:obj.n)
                x = np.arange(1, self.n+1) - np.floor(self.n/2+1);
            else:
                lower = -self.filter.sampleFactor * np.ceil(K_new)
                upper = np.ceil(K_new) * self.filter.sampleFactor
                x = np.arange(lower, upper+1)

            if s_hat != 0:
                f_hat = np.exp( -0.5 * (x/s_hat)**2 )
                f_hat = fftpack.ifftshift(f_hat)
            else:
                f_hat = 1


            f_hat = np.broadcast_to(f_hat, (1,1,f_hat.shape[0]) )
            a_hat = fftpack.fft(self.a.real, axis=2)
            # % This reduces the number of coefficients in the Fourier domain
            # % Beware of the normalization factor, 1/n, done by ifft

            if normalize < 2:
                raise NotImplementedError("Whatever crazy construction this is, "
                                          "I did not implement it yet.")
                #a_hat = a_hat(:,:,[1:ceil(K_new)*obj.filter.sampleFactor+1 end-ceil(K_new)*obj.filter.sampleFactor+1:end]);

            a_hat = a_hat * f_hat
            filter_new = OrientationSpaceFilter.OrientationSpaceFilter(
                                                            self.filter.f_c,
                                                            self.filter.b_f,
                                                            K_new)

            # % Consider using fft rather than ifft so that the mean is
            # % consistent
            if normalize == 1:
                Response = OrientationSpaceResponse(
                            filter_new,
                            fftpack.ifft(a_hat*a_hat.shape[2] / self.a.shape[2],
                                         axis=2)
                            )
            else:
                Response = OrientationSpaceResponse(filter_new,
                                                    fftpack.ifft(a_hat, axis=2)
                                                    )
            return Response
