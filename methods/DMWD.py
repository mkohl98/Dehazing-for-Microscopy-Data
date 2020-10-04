import tifffile as tf
import matplotlib.pyplot as plt
import pywt
import numpy as np
import scipy.stats as st
from skimage import restoration
import math
from skimage.measure import compare_ssim


class WaveletDehaze(object):
    def __init__(self, wavelet='haar'):
        self.wavelet = wavelet  # to have an overview about all possible wavelets: print(pywt.wavelist())
        self.original_image = np.array([])
        self.dehazed_image = np.array([])
        self.level = 5
        self.mode = 'constant'  # padding mode, also possible: zero, symmetric, asymmetric, ...

    def set_wavelet(self, wavelet):
        self.wavelet = wavelet

    def set_mode(self, mode):
        self.mode = mode

    def return_image_dehazed(self):
        return self.dehazed_image

    @staticmethod
    def __make_arr_quadratic(arr):
        arr_ = np.array(arr)
        min_dim = min(np.shape(arr_))
        while np.shape(arr_) != (min_dim, min_dim):
            while np.shape(arr_)[1] != min_dim:
                arr_ = arr_[:, :-1]
            while np.shape(arr_)[0] != min_dim:
                arr_ = arr_[:-1, :]
        return arr_

    @staticmethod
    def __make_array_fitting(arr1, arr_given):
        # shape arr_given < shape arr1
        arr1 = np.array(arr1)
        arr_given = np.array(arr_given)
        while np.shape(arr1) != np.shape(arr_given):
            # make arr1 symmetric by removing last row/column
            if (0, 0) == (np.shape(arr1)[0] % 2, np.shape(arr1)[1] % 2):
                pass
            elif (0, 1) == (np.shape(arr1)[0] % 2, np.shape(arr1)[1] % 2):
                arr1 = arr1[:, :-1]
            elif (1, 0) == (np.shape(arr1)[0] % 2, np.shape(arr1)[1] % 2):
                arr1 = arr1[:-1, :]
            elif (1, 1) == (np.shape(arr1)[0] % 2, np.shape(arr1)[1] % 2):
                arr1 = arr1[:-1, :-1]
            else:
                raise ValueError('Error404: This should not happen!')
            while np.shape(arr1) != np.shape(arr_given):
                # removes row and column
                arr1 = arr1[:-1, :-1]
        return arr1

    @staticmethod
    def __unmask(arr1, arr_2, threshold, substitute=0):
        arr1 = np.array(arr1)
        arr_2 = np.array(arr_2)
        arr_unmasked = arr_2
        if np.shape(arr1)[0] != np.shape(arr_unmasked)[0] or \
                np.shape(arr1)[1] != np.shape(arr_unmasked)[1]:
            arr1 = WaveletDehaze.__make_array_fitting(arr_unmasked, arr_unmasked)
        else:
            pass
        # threshold
        for x in range(len(arr1)):
            for y in range(len(arr1[0])):
                if arr1[x][y] < threshold:
                    arr_unmasked[x][y] = substitute
                else:
                    pass
        return arr_unmasked

    def dehaze(self, image, slide=0, level=5, threshold=10, substitute=1):
        '''
        Method to dehaze image out of imagestack (Tiff-File) by using Wavelet decomposition
        :param image: Tiff-File
        :param slide: int, index of slide of choice of image stack
        :param level: int, number of level of deconposition
        :param threshold: float, threshold of original image(!) which unmasks area on dehazed image
        :param substitute: float, substitute for thresholding
        :return: 2d array, return dehazed image with WaveletDehaze.return_image_dehazed()
        '''

        self.level = level
        # read image
        img_raw_ = tf.imread(image, key=slide)
        img_raw = WaveletDehaze.__make_arr_quadratic(img_raw_)
        self.original_image = img_raw_

        # pre-process input image
        if (0, 0) == (np.shape(img_raw)[0] % 2, np.shape(img_raw)[1] % 2):
            img = img_raw
        elif (0, 1) == (np.shape(img_raw)[0] % 2, np.shape(img_raw)[1] % 2):
            img = img_raw[:, :-1]
        elif (1, 0) == (np.shape(img_raw)[0] % 2, np.shape(img_raw)[1] % 2):
            img = img_raw[:-1, :]
        elif (1, 1) == (np.shape(img_raw)[0] % 2, np.shape(img_raw)[1] % 2):
            img = img_raw[:-1, :-1]
        else:
            raise ValueError('Error404: This should not happen!')

        # mean_w(f(t)) = mean_w(img)
        coeffs_raw = pywt.wavedec2(img, self.wavelet, self.mode, level=self.level)
        low_freq = coeffs_raw[0]
        coeffs_low = [low_freq]
        for x in range(self.level):
            coeffs_low.append((None, None, None))
        mean_w_img = pywt.waverec2(coeffs_low, self.wavelet, self.mode)

        # calculate difference d(t)
        difference = np.subtract(img, WaveletDehaze.__make_array_fitting(mean_w_img, img))

        # mean_w(d(t))^2
        coeffs_d_raw = pywt.wavedec2(difference, self.wavelet, self.mode, level=self.level)
        low_freq_d = coeffs_d_raw[0]
        coeffs_d_low = [low_freq_d]
        for x in range(self.level):
            coeffs_d_low.append((None, None, None))
        mean_w_d = pywt.waverec2(coeffs_d_low, self.wavelet, self.mode)

        # calculate a(t)
        a_t = np.sqrt(np.multiply(2, np.square(mean_w_d)))

        # calculate img_rec = e(t)
        img_rec = np.add(difference, WaveletDehaze.__make_array_fitting(a_t, difference))

        # unmask reconstructed image
        self.dehazed_image = WaveletDehaze.__unmask(self.original_image, img_rec, threshold=threshold,
                                                    substitute=substitute)

    def plot(self, comparative_image_stack, slide):
        '''
        Plot dehazed and another comparative image
        :param comparative_image_stack: Tiff-file
        :param slide: int, number of slide of comaparative_image
        :return: Plot of dehazed and comparative image
        '''
        comp = tf.imread(comparative_image_stack, key=slide)
        fig = plt.figure(figsize=(12, 5))
        titles = ['Original image', 'Dehazed image', 'Ground truth image']
        for i, a in enumerate([self.original_image, self.dehazed_image, comp]):
            ax = fig.add_subplot(1, len(titles), i + 1)
            ax.imshow(a, interpolation="nearest", vmin=0, cmap=plt.cm.gray)
            ax.set_title(titles[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()

    def deconvolve(self, psf, iterations):
        '''
        Method for deconvolution of convolved 2d data by using Richardson-Lucy algorithm
        :param psf: tuple(float(Gaussian kernel length), float(Gaussian kernel sigma))
                    2d array, point spread function
        :param iterations: int, number of iterations
        :return: 2d array, convolved data
        '''
        def gkern(kernlen=psf[0], nsig=psf[1]):
            """Returns a 2D Gaussian kernel array."""

            interval = (2 * nsig + 1.) / (kernlen)
            x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
            kern1d = np.diff(st.norm.cdf(x))
            kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
            kernel = kernel_raw / kernel_raw.sum()
            return kernel
        if type(psf) == tuple:
            current_psf = gkern()
        else:
            current_psf = psf
        self.dehazed_image = restoration.richardson_lucy(self.dehazed_image, current_psf, iterations=iterations,
                                                         clip=False)

    @staticmethod
    def basic_psnr(img1, img2, max_value=65535.0):
        # basic implementation of psnr calculation
        # max_value = 255 for 8bit, 65535 for 16bit, 4294967295 for 32bit
        mse = ((img1 - img2) ** 2).mean()
        if mse == 0:
            return 'identical'
        else:
            return 20 * math.log10(max_value / math.sqrt(mse))

    @staticmethod
    def best_psnr(gt, pred):
        # alternative method which finds the best fitting psnr
        def fix_range(gt, x):
            a = np.sum(gt * x) / (np.sum(x * x))
            # print(a)
            return x * a

        def fix(gt, x):
            return fix_range(gt - np.mean(gt), x - np.mean(x))

        def psnr(gt, pred, range_=255.0):
            mse = np.mean((gt - pred) ** 2)
            return 20 * np.log10((range_) / np.sqrt(mse))

        gt_ = (gt - np.mean(gt)) / np.std(gt)
        ra = (np.max(gt_) - np.min(gt_))
        return psnr(gt_, fix(gt_, pred), ra)

    @staticmethod
    def ssim(img1, img2):
        return compare_ssim(img1, img2)

    def plot_wavelet_x_level(self, imagestack, slide, waveletlist, levellist, threshold=10, substitute=1):
        '''
        Plots dehazed image by wavelet(x-axis) and level(y-axis)
        :param imagestack: Tiff-file
        :param slide: int, index of slide of choice of image stack
        :param waveletlist: list of strings of wavelets,
                            to have an overview about all possible wavelets: print(pywt.waveletlist())
        :param levellist: list of ints, containing level of decomposition
        :param threshold: float, threshold of original image(!) which unmasks area on dehazed image
        :param substitute: float, substitute for thresholding
        '''
        out = []

        for lvl in levellist:
            for wavelet in waveletlist:
                WaveletDehaze.set_wavelet(self, wavelet)
                WaveletDehaze.dehaze(self=self, image=imagestack, slide=slide, level=lvl, threshold=threshold,
                                     substitute=substitute)
                out.append(self.dehazed_image)

        counter = 0
        for lvl in levellist:
            for wavelet in waveletlist:
                counter += 1
                plt.axis("off")
                plt.subplot(len(levellist), len(waveletlist), counter)
                plt.title('level:{}, Wavelet:{}'.format(lvl, wavelet))
                plt.imshow(out[counter-1], interpolation="nearest", vmin=0, cmap=plt.cm.gray)
        plt.axis("off")
        plt.tight_layout()
        plt.show()