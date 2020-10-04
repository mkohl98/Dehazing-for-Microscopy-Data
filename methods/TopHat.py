import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf
import cv2
from Methods import Metrics


def threshold(im, value, substitute=0):
    for x in range(len(im)):
        for y in range(len(im[0])):
            if im[x][y] < value:
                im[x][y] = substitute
    return im


def grid_search(deg_, gt, array_win, array_th, kerneltype='ellipse', metric='SSIM'):

    grid = [[[None, 0] for y in array_win] for x in array_th]

    for th in range(len(array_th)):
        for win in range(len(array_win)):

            if kerneltype is 'ellipse':
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (array_win[win], array_win[win]))
            elif kerneltype is 'cube':
                kernel = np.ones(shape=(array_win[win], array_win[win]))
            else:
                raise ValueError('kerneltype needs to be str. ellipse ore cube')
            top_ = cv2.morphologyEx(deg_, cv2.MORPH_TOPHAT, kernel)
            top = np.array(threshold(top_, array_th[th]), dtype=np.float32)

            grid[th][win][0] = list(top)
            if metric is 'SSIM':
                grid[th][win][1] = Metrics.SSIM(top, gt)
            elif metric is 'PSNR':
                grid[th][win][1] = Metrics.best_PSNR(deg_, top)
            else:
                raise ValueError('metric needs to be str. SSIM or PSNR')
    return grid


def get_best(arr_res, arr_th, arr_win, raw=None, gt=None, plot=False):
    best_value = 0
    best_image = []
    best_params = []
    for th in range(len(arr_th)):
        for win in range(len(arr_win)):
            if arr_res[th][win][1] < best_value:
                pass
            else:
                best_value = arr_res[th][win][1]
                # best_image = arr_res[th][win][1]
                best_params = [arr_th[th], arr_win[win]]
    print('best window size: {}'.format(best_params[1]))
    print('best threshold: {}'.format(best_params[0]))
    print('best value: {}'.format(best_value))

    if plot is True:
        fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
        ax[0].set_title('Raw Image')
        ax[0].imshow(raw, cmap='magma')
        ax[0].axis('off')
        ax[1].set_title('Top Hat Prediction')
        ax[1].imshow(arr_res[arr_th.index(best_params[0])][arr_win.index(best_params[1])][0], cmap='magma')
        ax[1].axis('off')
        ax[2].set_title('Ground Truth')
        ax[2].imshow(gt, cmap='magma')
        ax[2].axis('off')
        plt.show()
    best_image = arr_res[arr_th.index(best_params[0])][arr_win.index(best_params[1])][0]
    return best_value, best_image, best_params


def multi_grid_search(deg_tiff, gt_tiff, depth, arr_th, arr_win, kerneltype='ellipse', metric='SSIM'):
    # return 3d image stack with dimension (depth,x,y) containing top hat results
    stack = []
    SSIM_sum = 0

    for slide in range(depth):

        im_deg = tf.imread(deg_tiff, key=slide)
        im_gt = tf.imread(gt_tiff, key=slide)
        recent_grid = grid_search(deg_=im_deg, gt=im_gt, array_win=arr_win, array_th=arr_th, kerneltype=kerneltype,
                                  metric=metric)
        recent_pred = get_best(arr_res=recent_grid, arr_th=arr_th, arr_win=arr_win, )
        stack.append(recent_pred[1])
        SSIM_sum += recent_pred[0]

    SSIM_average = SSIM_sum/depth
    print('average maximizing value = {}'.format(SSIM_average))
    return stack


def write_tiff(array, name):
    array_ = np.array(array).astype(np.float32)
    tf.imwrite(file=name, data=array_, photometric='minisblack')


# ---- do something here -------------------------------


thresh = []  # array containing threshold values
window = []  # array containing window size values

img = tf.imread('image.tif', key=0)
gt = np.array(tf.imread('gr.tif', key=0))
results = grid_search(img, gt, window, thresh, metric='PSNR')
x = get_best(results, thresh, window, img, gt, True)
