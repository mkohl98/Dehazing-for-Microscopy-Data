import numpy as np
import random
import matplotlib.pyplot as plt
import math
from skimage.draw import bezier_curve
from copy import deepcopy
from scipy.ndimage import gaussian_filter as gauss
from pathlib import Path
from tifffile import imwrite as tw


class Arc(object):
    @staticmethod
    def draw_arc(min_angle, max_angle, array_shape, intensity=255, weight=0.4, directed=None):
        '''
        Method which draws an Arc.
        :param min_angle: int, minimal angle for curve generation
        :param max_angle: int, maximum angle for curve generation
        :param array_shape: tuple(int, int), shape of image
        :param intensity: int, set intensity of arc
        :param weight: float, middle control point weight, tension of line
        :param directed: tuple(center point area1, center point area2, radius)
        :return: 2d array, containing bezier curve like arc
        '''
        gamma = random.randint(min_angle, max_angle)
        arr = np.zeros(array_shape, dtype=np.uint8)
        length = array_shape[1]
        width = array_shape[0]

        if directed is None:
            p1 = (random.randint(0, width), random.randint(0, length))
            p2 = (random.randint(0, width), random.randint(0, length))
        else:
            p1 = (random.randint(directed[0][0] - directed[2], directed[0][0] + directed[2]),
                  random.randint(directed[0][1] - directed[2], directed[0][1] + directed[2]))
            p2 = (random.randint(directed[1][0] - directed[2], directed[1][0] + directed[2]),
                  random.randint(directed[1][1] - directed[2], directed[1][1] + directed[2]))

        alpha = (180 - gamma) / 2
        p1p2 = math.sqrt(np.square(p2[0] - p1[0]) + (np.square(p2[1] - p1[1])))

        p1pm = math.sin(alpha) / math.sin(gamma) * p1p2

        x_p3 = int(abs(p1pm * math.cos(alpha)))
        y_p3 = int(abs(p1pm * math.sin(alpha)))

        # draw bezier curve
        rr, cc = bezier_curve(p1[0], p1[1], x_p3, y_p3, p2[0], p2[1], weight, array_shape)
        arr[rr, cc] = intensity
        return arr

    @staticmethod
    def thickness(array, line_width, substitute=255):
        for width in range(line_width):
            for x in range(len(array)):
                for y in range(len(array[0])):
                    if array[x][y] <= 0:
                        pass
                    else:
                        if x - width >= 0:
                            array[x - width][y] = substitute
                        if y - width >= 0:
                            array[x][y - width] = substitute
                        if x - width >= 0 and y - width >= 0:
                            array[x - width][y - width] = substitute
        return array


class ImagePair(Arc):
    def __init__(self, amount=1, seed=None):
        '''
        Constructor of class ImagePair
        :param amount: int, amount >= 1, amount of images to generated
                       NOTE: just change if you want to generate stacks of images, use gen_multiple_images()
        :param seed: seed for rng
        '''
        self.amount = amount  # amount of images to generate
        np.random.seed(seed)
        random.seed(seed)

        self.clean_image = []  # stores the clean image
        self.degraded_image = []  # stores the degraded image
        self.infocus_image = []

        self.image_shape = ()  # stores the shape of the images
        self.conv_strength = []  # list containing coefficients of strength of out of focus for each arc

        self.degraded_images = [[] for _ in range(self.amount)]  # list containing degraded images
        self.clean_images = [[] for _ in range(self.amount)]  # list containing clean images
        self.infocus_images = [[] for _ in range(self.amount)]

    def __del__(self):
        print('Instance deleted.')

    def return_clean_image(self):
        '''
        If there is just one CLEAN image generated by gen_images() this image gets returned
        '''
        return self.clean_image

    def return_infocus_image(self):
        '''
        If there is just one CLEAN INFOCUS image generated by gen_images() this image gets returned
        '''
        return self.infocus_image

    def return_degraded_image(self):
        '''
        If there is just one DEGRADED image generated by gen_images() this images gets returned
        '''
        return self.degraded_image

    def return_clean_image_n(self, n):
        '''
        Method to return a specified CLEAN image generated by gen_multiple_images
        :param n: int, index of clean image in image stack
        :return: 2d array, specified clean image
        '''
        return self.clean_images[n]

    def return_degraded_image_n(self, n):
        '''
        Method to return a specified DEGRADED image generated by gen_multiple_images
        :param n: int, index of degraded image in image stack
        :return: 2d array, specified degraded image
        '''
        return self.degraded_images[n]

    def return_infocus_image_n(self, n):
        '''
        Method to return a specified CLEAN INFOCUS image generated by gen_multiple_images
        :param n: int, index of infocus image in image stack
        :return: 2d array, specified infocus image
        '''
        return self.infocus_images[n]

    @staticmethod
    def __stack_to_img(stack, count, shape, overlapping=True):
        # if overlapping is True: default, intensities get summed
        # if overlapping is False, highest intensity get stored
        image = np.zeros(shape, dtype=np.uint8)
        if overlapping is True:
            for s in range(count):
                image = np.add(image, stack[s])
        elif overlapping is False:
            image = np.amax(stack, 0)
        else:
            raise TypeError('Arg overlapping must have type: boolean.')
        return image

    @staticmethod
    def __generate_perlin_noise_2d(shape, res):
        # private method to pre process perlin noise
        # implementation by https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html
        def f(t):
            return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        # Gradients
        angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
        # Interpolation
        t = f(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

    def gen_images(self, image_shape, min_angle, max_angle, line_width, curve_count, clean_curves=1, fadeout=1.0,
                   weight=0.5, overlapping=True, intensity=100, directed=None):
        '''
        Method which creates a corresponding pair of ground truth and degraded image.
        :param image_shape: tuple(int, int), contains shape of image
        :param min_angle: int, minimum angle for bezier arc generation
        :param max_angle: int, maximum angle for bezier arc generation
        :param line_width: int, thickness of line of arc
        :param curve_count: int, number of curves in final image
        :param clean_curves: int, number of curves which have not to be convolved
        :param fadeout: float: level for Gaussian filter
        :param weight: float, middle control point weight, tension of arc
        :param overlapping: overlapping structures intensity will not get summed
        :param intensity: int in range 0-255, optional, intensity of curves
        :param directed: int, radius of size of starting and destination patch
        :return: ground truth image(2d array), degraded image (2d array), ground trouth infocus image(2d array)
        '''

        # draw arcs, generate stacks
        self.image_shape = image_shape
        clean_img_stack = np.ndarray((curve_count, image_shape[0], image_shape[1]))
        if directed is None:
            for curve in range(curve_count):
                clean_img_stack[curve] = Arc.draw_arc(min_angle=min_angle, max_angle=max_angle, array_shape=image_shape,
                                                      intensity=intensity, weight=weight)
                clean_img_stack[curve] = Arc.thickness(clean_img_stack[curve], line_width=line_width,
                                                       substitute=intensity)
        else:
            center1 = (random.randint(directed, self.image_shape[0] - directed),
                       random.randint(directed, self.image_shape[1] - directed))
            center2 = (random.randint(directed, self.image_shape[0] - directed),
                       random.randint(directed, self.image_shape[1] - directed))
            for curve in range(curve_count):
                clean_img_stack[curve] = Arc.draw_arc(min_angle=min_angle, max_angle=max_angle, array_shape=image_shape,
                                                      intensity=intensity, weight=weight,
                                                      directed=(center1, center2, directed))
                clean_img_stack[curve] = Arc.thickness(clean_img_stack[curve], line_width=line_width,
                                                       substitute=intensity)
        convolved_img_stack = deepcopy(clean_img_stack)

        # create convolution strength levels
        for curve_ in range(curve_count):
            self.conv_strength.append(random.randint(2, 10))
        for clean in range(clean_curves):
            self.conv_strength[clean] = random.uniform(0.1, 1)

        # convolve curves
        for curve in range(curve_count):
            convolved_img_stack[curve] = gauss(convolved_img_stack[curve], fadeout * self.conv_strength[curve])

        # generate clean in-focus image
        self.infocus_image = np.zeros((self.image_shape))
        for clean in range(clean_curves):
            self.infocus_image += clean_img_stack[clean]

        # turn stacks into images
        self.clean_image = ImagePair.__stack_to_img(clean_img_stack, count=curve_count,
                                                    shape=image_shape, overlapping=overlapping)
        self.degraded_image = ImagePair.__stack_to_img(convolved_img_stack, count=curve_count,
                                                       shape=image_shape, overlapping=overlapping)

    def gen_multiple_images(self, image_shape, min_angle, max_angle, line_width, curve_count, clean_curves=1, fadeout=1,
                            weight=0.5, overlapping=True, intensity=100, directed=None):
        '''
        This method creates stacks of clean, in-focus and corresponding degraded images if amount in constructor is > 1.
        :param image_shape: tuple(int, int), contains shape of image
        :param min_angle: int, minimum angle for bezier arc generation
        :param max_angle: int, maximum angle for bezier arc generation
        :param line_width: int, thickness of line of arc
        :param curve_count: int, number of curves in final image
        :param clean_curves: int, number of curves which have not to be convolved
        :param fadeout: float: level for Gaussian filter
        :param weight: float, middle control point weight, tension of arc
        :param overlapping: overlapping structures intensity will not get summed
        :param intensity: int in range 0-255, optional, intensity of curves
        :param directed: int, radius of size of starting and destination patch
        :return: creates ground truth image stack (3d array), degraded image stack (3d array)
        '''
        self.image_shape = image_shape
        for stack in range(self.amount):
            # clear existing images
            self.clean_image = []
            self.degraded_image = []
            self.infocus_image = []
            ImagePair.gen_images(self=self, image_shape=self.image_shape, min_angle=min_angle, max_angle=max_angle,
                                 line_width=line_width, curve_count=curve_count, clean_curves=clean_curves,
                                 fadeout=fadeout, weight=weight, overlapping=overlapping,
                                 intensity=intensity, directed=directed)
            self.degraded_images[stack] = ImagePair.return_degraded_image(self=self)
            self.clean_images[stack] = ImagePair.return_clean_image(self=self)
            self.infocus_images[stack] = ImagePair.return_infocus_image(self=self)

    def offset(self, value):
        if self.amount == 1:
            self.degraded_image = self.degraded_image + value
        else:
            for stack in range(self.amount):
                self.degraded_images[stack] = self.degraded_images[stack] + value

    def set_gauss_noise(self, gmean=0.0, gstd=1.0, lazy=True):
        '''
        Method which sets Gaussian white noise at degraded image.
        :param gmean: float, Gaussian mean
        :param gstd: float, Gaussian standard deviation
        :param lazy: bool, if True: noise is computed only one time and applied to all images,
                           if False: noise is separately computed for each image
        '''
        if self.amount == 1:
            noise = np.array([np.random.normal(gmean, gstd, self.image_shape[0]) for x in range(self.image_shape[1])])
            noise = abs(noise)  # absolute values to avoid negative values by adding them to zero intensity
            self.degraded_image = np.add(self.degraded_image, noise)
        else:
            if lazy is True:
                # generation of noise
                noise = np.array(
                    [np.random.normal(gmean, gstd, self.image_shape[0]) for x in range(self.image_shape[1])])
                noise = abs(noise)

                # add noise to images
                for stack in range(self.amount):
                    self.degraded_images[stack] = self.degraded_images[stack] + noise
            else:
                for stack in range(self.amount):
                    # generate noise for each image
                    noise = np.array(
                        [np.random.normal(gmean, gstd, self.image_shape[0]) for x in range(self.image_shape[1])])
                    noise = abs(noise)
                    # add noise to image
                    self.degraded_images[stack] = self.degraded_images[stack] + noise

    def set_poisson_noise(self, factor=1):
        if self.amount == 1:
            noise = np.random.poisson(self.degraded_image) - self.degraded_image
            factorized_noise = noise * factor
            self.degraded_image = self.degraded_image + factorized_noise
        else:
            for stack in range(self.amount):
                noise = np.random.poisson(self.degraded_images[stack]) - self.degraded_images[stack]
                factorized_noise = noise * factor
                self.degraded_images[stack] = self.degraded_images[stack] + factorized_noise

    def set_pearlin_noise(self, vmin=0, vmax=30, res=(8, 8), lazy=True):
        if self.amount == 1:
            # generate noise
            noise = ImagePair.__generate_perlin_noise_2d(self.image_shape, res=res)
            # rescale noise
            noise = noise + 1 + vmin
            noise_rescale = noise / 2 * (vmax - vmin)
            # add noise to degraded image
            self.degraded_image = self.degraded_image + noise_rescale
        else:
            if lazy is True:
                # generate noise
                noise = ImagePair.__generate_perlin_noise_2d(self.image_shape, res=res)
                # rescale noise
                noise = noise + 1 + vmin
                noise_rescale = noise / 2 * (vmax - vmin)
                for stack in range(self.amount):
                    self.degraded_images[stack] = self.degraded_images[stack] + noise_rescale
            else:
                for stack in range(self.amount):
                    # generate noise
                    noise = ImagePair.__generate_perlin_noise_2d(self.image_shape, res=res)
                    # rescale noise
                    noise = noise + 1 + vmin
                    noise_rescale = noise / 2 * (vmax - vmin)
                    # add noise to degraded image
                    self.degraded_images[stack] = self.degraded_images[stack] + noise_rescale

    def set_fractal_noise(self, vmin, vmax, res=(8, 8), octaves=1, persistent=0.5, lazy=True):
        def generate_fractal_noise_2d(shape, _res, _octaves, persistence):
            # implementation by https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html
            noise = np.zeros(shape)
            frequency = 1
            amplitude = 1
            for _ in range(_octaves):
                noise += amplitude * ImagePair.__generate_perlin_noise_2d(shape,
                                                                          (frequency * _res[0], frequency * _res[1]))
                frequency *= 2
                amplitude *= persistence
            return noise

        if self.amount == 1:
            # generate noise
            fractal_noise = generate_fractal_noise_2d(self.image_shape, _res=res, _octaves=octaves,
                                                      persistence=persistent)
            # rescale noise
            fractal_noise = fractal_noise + abs(np.amin(fractal_noise)) + vmin
            fractal_noise_rescale = fractal_noise / (np.amax(fractal_noise) - np.amin(fractal_noise)) * (vmax - vmin)
            # add noise
            self.degraded_image = self.degraded_image + fractal_noise_rescale
        else:
            if lazy is True:
                # generate noise
                fractal_noise = generate_fractal_noise_2d(self.image_shape, _res=res, _octaves=octaves,
                                                          persistence=persistent)
                # rescale noise
                fractal_noise = fractal_noise + abs(np.amin(fractal_noise)) + vmin
                fractal_noise_rescale = fractal_noise / (np.amax(fractal_noise) - np.amin(fractal_noise)) * (
                        vmax - vmin)
                for stack in range(self.amount):
                    self.degraded_images[stack] = self.degraded_images[stack] + fractal_noise_rescale
            else:
                for stack in range(self.amount):
                    # generate noise
                    fractal_noise = generate_fractal_noise_2d(self.image_shape, _res=res, _octaves=octaves,
                                                              persistence=persistent)
                    # rescale noise
                    fractal_noise = fractal_noise + abs(np.amin(fractal_noise)) + vmin
                    fractal_noise_rescale = fractal_noise / (np.amax(fractal_noise) - np.amin(fractal_noise)) * (
                            vmax - vmin)
                    self.degraded_images[stack] = self.degraded_images[stack] + fractal_noise_rescale

    def plot(self, nums=None):
        '''
        This Method plots corresponding images. If there are more than one pair of images is generated specify the
        indices of wanted images in param nums.
        :param nums: list, indices of wanted images
        :return: plot of images
        '''
        if self.amount == 1:
            fig = plt.figure(figsize=(11, 5))
            titles = ['Degraded image', 'Ground trouth image', 'Ground trouth in-focus image']
            for i, a in enumerate([self.degraded_image, self.clean_image, self.infocus_image]):
                ax = fig.add_subplot(1, len(titles), i + 1)
                ax.imshow(a, interpolation="nearest", vmin=0, cmap=plt.cm.gray)  # vmax=255, # cmap=plt.cm.gray
                ax.set_title(titles[i], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            plt.show()
        else:
            fig = plt.figure(figsize=(10, 5 * len(nums)))
            counter = 0
            versions = ['Degraded image', 'Ground trouth image', 'Ground trouth in-focus image']
            for num in range(len(nums)):
                for version in range(len(versions)):
                    counter += 1
                    plt.axis("off")
                    plt.subplot(len(nums), 3, counter)
                    plt.title('{} no.:{}'.format(versions[version], nums[num]))
                    if version == 0:
                        plt.imshow(self.degraded_images[nums[num]], interpolation="nearest", vmin=0, cmap=plt.cm.gray)
                    elif version == 1:
                        plt.imshow(self.clean_images[nums[num]], interpolation="nearest", vmin=0, cmap=plt.cm.gray)
                    elif version == 2:
                        plt.imshow(self.infocus_images[nums[num]], interpolation="nearest", vmin=0, cmap=plt.cm.gray)
            plt.tight_layout()
            plt.axis("off")
            plt.show()

    def save(self, path='./new_dir', name=None):
        '''
        This method saves gen_images generated images as png ans gen_multiple_images generated images as tif.
        :param path: string, path to directory, it will be generated if it is not existing
        :param name: string, optional name as file ending
        '''
        if self.amount == 1:
            Path(path).mkdir(parents=True, exist_ok=True)
            if name is None:
                path_clean = path + '/ground_truth_image.png'
                path_degraded = path + '/degraded_image.png'
                path_infocus = path + '/infocus_image.png'
            else:
                path_clean = path + '/ground_truth_image_{}.png'.format(name)
                path_degraded = path + '/degraded_image_{}.png'.format(name)
                path_infocus = path + '/infocus_image_{}.png'.format(name)
            plt.imsave(path_clean, self.clean_image, vmin=0, cmap=plt.cm.gray)
            plt.imsave(path_degraded, self.degraded_image, vmin=0, cmap=plt.cm.gray)
            plt.imsave(path_infocus, self.infocus_image, vmin=0, cmap=plt.cm.gray)
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            # pre process data type
            self.degraded_images = np.array(self.degraded_images).astype(np.float32)
            self.clean_images = np.array(self.clean_images).astype(np.float32)
            self.infocus_images = np.array(self.infocus_images).astype(np.float32)
            # change name if needed
            if name is None:
                path_clean = path + '/ground_truth_images.tif'
                path_degraded = path + '/degraded_images.tif'
                path_infocus = path + '/infocus_image.tif'
            else:
                path_clean = path + '/ground_truth_images_{}.tif'.format(name)
                path_degraded = path + '/degraded_images_{}.tif'.format(name)
                path_infocus = path + '/infocus_images_{}.tif'.format(name)
            tw(path_clean, self.clean_images, photometric='minisblack')
            tw(path_degraded, self.degraded_images, photometric='minisblack')
            tw(path_infocus, self.infocus_images, photometric='minisblack')

        '''
        ### example for one image pair ###

        > example_one = ImagePair(seed=43)
            # create instance of object ImagePair WITHOUT changing parameter amount
        > example_one.gen_images(image_shape=(256, 256), min_angle=35, max_angle=100,
                  line_width=2, curve_count=15, clean_curves=4, fadeout=0.5, weight=0.1
                  intensity=100, overlapping=True, directed=20)
            # generate two images, ground truth and degraded, stored in class variables

        > example_one.offset(10)
            # create offset of 10 of degraded image as optional pre processing step for Poisson noise
        > example_one.set_poisson_noise(3)
            # adds poisson noise
        > example_one.set_gauss_noise(10, 6)
            # adds Gaussian white noise to the degraded image with Gaussian mean = 6 and Gaussian standard deviation = 3

        > example_one.set_fractal_noise(0, 30, (4, 4), 2, 0.5)
            # adds out-of-focus light

        > example_one.plot()
            # plots the two corresponding images
        > example_one.save(name='example_one')
            # saves the two images in new directory ./new_dir (default) as png
        ________________________________________________________________________________________________________________

        ### example for multiple image pairs ###

        > example_two = ImagePair(seed=42, amount=15)
            # create instance of ImagePair with 15 image pairs

        > example_two.gen_multiple_images(image_shape=(256, 256), min_angle=35, max_angle=100,
                                          line_width=2, curve_count=15, clean_curves=4, fadeout=0.5, weight=0.1
                                          intensity=100, overlapping=True, directed=20)
            # create 15 image pairs like in example_one above

        > example_two.offset(30)

        > example_two.set_poisson_noise(1.5)

        > example_two.set_gauss_noise(20, 13)

        > example_two.set_fractal_noise(0, 60, (8, 8), 2, 0.5, lazy=True)

        > example_two.plot([0, 5, 9])
            # plots image pairs 0, 5 and 9

        > example_two.save(path='./new_dir', name='TEST2')
            # saves ground truth and degraded images as ground_truth_images_TEST2.tif and degraded_images_TEST2.tif
'''
