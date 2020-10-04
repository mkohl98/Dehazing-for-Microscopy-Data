''' Metrics for image analysis '''
import numpy as np


def best_PSNR(gt, pred_):
    # alternative method which finds the best fitting psnr
    def fix_range(gt, x):
        a = np.sum(gt * x) / (np.sum(x * x))
        # print(a)
        return x * a

    def fix(gt, x):
        return fix_range(gt - np.mean(gt), x - np.mean(x))

    def psnr(gt, pred__, range_=255.0):
        mse = np.mean((gt - pred__) ** 2)
        return 20 * np.log10((range_) / np.sqrt(mse))

    gt_ = (gt - np.mean(gt)) / np.std(gt)
    ra = (np.max(gt_) - np.min(gt_))
    return psnr(gt_, fix(gt_, pred_), ra)


def basic_PSNR(gt, pred__, range_=255.0):
    mse = np.mean((gt - pred__) ** 2)
    return 20 * np.log10((range_) / np.sqrt(mse))


def NPSNR(gt, pred__, range_=255.0):
    # normalized PSNR; Normalization through dynamic range of predicted image
    mse = np.mean((gt - pred__) ** 2)
    return 20 * np.log10((range_) / np.sqrt(mse/((np.amax(pred__)) - np.amin(np.stack(pred__)))))


# def alternative_SSIM(imx, imy):
#     # Source: https://en.wikipedia.org/wiki/Structural_similarity
#     mean_x = np.mean(imx)
#     mean_y = np.mean(imy)
#     std_x = np.std(imx)
#     std_y = np.std(imy)
#     L = np.amax(np.stack((imy, imy))) - np.amin(np.stack((imx, imy)))
#     c1 = (0.01 * L) ** 2
#     c2 = (0.03 * L) ** 2
#     cov_xy = np.mean((imx-mean_x)*(imy-mean_y))
#
#     ssim1 = (2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)
#     ssim2 = (mean_x ** 2 + mean_y ** 2 + c1) * (std_x ** 2 + std_y ** 2 + c2)
#     return np.array(ssim1 / ssim2).astype(np.float16)


def SSIM(imx, imy):
    # Literature: Wang et al., "Image Quality Assessement: From Error Visibility to Structural Similarity", 2004
    mean_x = np.mean(imx)
    mean_y = np.mean(imy)
    std_x = np.std(imx)
    std_y = np.std(imy)
    lrange = np.amax(np.stack((imy, imy))) - np.amin(np.stack((imx, imy)))
    c1 = (0.01 * lrange) ** 2
    c2 = (0.03 * lrange) ** 2
    cov_xy = np.sum((imx-mean_x)*(imy-mean_y)) * (1/(len(imx) * len(imx[0]) - 1))

    ssim1 = (2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)
    ssim2 = (mean_x ** 2 + mean_y ** 2 + c1) * (std_x ** 2 + std_y ** 2 + c2)
    return np.array(ssim1 / ssim2).astype(np.float16)


def adjustable_SSIM(imx, imy, alpha=1, beta=1, gamma=1):
    # Literature: Wang et al., "Image Quality Assessement: From Error Visibility to Structural Similarity", 2004
    # alpha adjusts luminance measure
    # beta adjusts contrast measure
    # gamma adjusts similarity measure

    if 1 < (alpha or beta or gamma) < 0:
        raise ValueError('Alpha, Beta and Gamma needs to be between 1 and 0.')

    mean_x = np.mean(imx)
    mean_y = np.mean(imy)
    std_x = np.std(imx)
    std_y = np.std(imy)
    lrange = np.amax(np.stack((imy, imy))) - np.amin(np.stack((imx, imy)))
    c1 = (0.01 * lrange) ** 2
    c2 = (0.03 * lrange) ** 2
    c3 = c2/2
    cov_xy = np.sum((imx - mean_x) * (imy - mean_y)) * (1 / (len(imx) * len(imx[0]) - 1))

    luminance = (2 * mean_x * mean_y + c1)/(mean_x**2 + mean_y**2 + c1)
    contrast = (2 * std_x * std_y + c2)/((std_x**2) + (std_y**2) + c2)
    similarity = (cov_xy + c3)/(std_x * std_y + c3)

    return (luminance ** alpha) * (contrast ** beta) * (similarity ** gamma)


def MSE(im1, im2):
    # mean squared error
    return np.mean(np.subtract(im1, im2)**2)


def NMSE(im1, im2):
    return MSE(im1, im2)/((np.amax(im2)) - np.amin(np.stack(im2)))


def RMSE(im1, im2):
    # root mean squared error
    return MSE(im1, im2) ** 0.5


def NRMSE(im1, im2):
    # normalized root mean squared error
    return RMSE(im1, im2)/(np.amax(np.stack((im1, im2))) - np.amin(np.stack((im1, im2))))
