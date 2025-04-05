import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy
import random

def hadamard_matrix(n):
    if (n & (n - 1)) != 0 or n <= 0:
        raise ValueError("n must be a power of 2")

    H = np.array([[1]])

    while H.shape[0] < n:
        H = np.block([[H, H],
                      [H, -H]])
        
    H = np.where(H < -1, 0, 1)
    return H


def Combined_loss(generated, original):
    mse_function = nn.MSELoss()
    mse_l = mse_function(generated, original)
    psnr_l = PSNR(generated, original)
    # ssim_l = SSIM(generated, original, 64)
    return mse_l+psnr_l#+ssim_l


def gaussian_kernel(window_size, sigma=1.5):
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()  # Normalize
        kernel = g.unsqueeze(0) * g.unsqueeze(1)  # Outer product to form 2D Gaussian kernel
        return kernel

def PSNR(img1, img2, max_pixel_value=1.0):
    mse = torch.mean((img1 - img2) ** 2)  # Mean Squared Error
    if mse == 0:  # Avoid division by zero
        return float('inf')
    psnr = 10 * torch.log10(max_pixel_value ** 2 / mse)
    return psnr.item()

def SSIM(img1, img2, in_shape, window_size = 11, size_average = True, max_pixel_value = 1.0):
    # print('ssim shapes: ', img1.shape, img2.shape)
    img1 = torch.reshape(img1, (in_shape, in_shape))
    img2 = torch.reshape(img2, (in_shape, in_shape))

    kernel = gaussian_kernel(window_size).to(img1.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, window_size, window_size)
    
    # Padding
    padding = window_size // 2
    img1 = F.pad(img1.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
    img2 = F.pad(img2.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='reflect')
    
    # Mean
    mu1 = F.conv2d(img1, kernel)
    mu2 = F.conv2d(img2, kernel)
    
    # Squares of means
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Variances and covariances
    sigma1_sq = F.conv2d(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel) - mu1_mu2
    
    # SSIM constants
    C1 = (0.01 * max_pixel_value) ** 2
    C2 = (0.03 * max_pixel_value) ** 2
    
    # Compute SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Return the mean SSIM value
    return ssim_map.mean().item()


def hamming_score(vector1, vector2):
    assert len(vector1) == len(vector2), "Vectors must be of the same length"

    vector1 = torch.tensor(vector1, dtype=torch.float32)
    vector2 = torch.tensor(vector2, dtype=torch.float32)
    if len(torch.unique(vector1) != 2):
        vector1 = torch.where(vector1 < 0, 0, 1)
        vector2 = torch.where(vector2 < 0, 0, 1)

    # Calculate the number of differing bits
    differing_bits = (vector1 != vector2).float().sum()
    # Calculate the Hamming score
    hamming_score = differing_bits / len(vector1)
    return hamming_score.item()


def IsScalar(x):
    if type(x) in (list, np.ndarray,):
        return False
    else:
        return True

def Thresh(x):
    if IsScalar(x):
        val = 1 if x>0 else -1
    else:
        val = np.ones_like(x)
        val[x<0] = -1.
    return val

def Hamming(x, y):
    '''
        d = Hamming(x,y)
        
        Hamming distance between two binary vectors x and y.
        It's the number of digits that differ.
        
        Inputs:
          x and y are arrays of binary vectors, and can be either {0,1} or {-1,1}
        
        Output:
          d is the number of places where the inputs differ
    '''
    d = []
    for xx, yy in zip(x,y):
        dd = 0.
        for xxx,yyy in zip(xx,yy):
            if xxx==1 and yyy!=1:
                dd += 1.
            elif yyy==1 and xxx!=1:
                dd += 1.
        d.append(dd)
    return d

def Perturb(x, p=0.1):
    '''
        y = Perturb(x, p=0.1)
        
        Apply binary noise to x. With probability p, each bit will be randomly
        set to -1 or 1.
        
        Inputs:
          x is an array of binary vectors of {-1,1}
          p is the probability of each bit being randomly flipped
        
        Output:
          y is an array of binary vectors of {-1,1}
    '''
    y = copy.deepcopy(x)
    for yy in y:
        for k in range(len(yy)):
            if np.random.rand()<p:
                yy[k] = Thresh(np.random.randint(2)*2-1)
    return y


def perturb_pattern(image, perturb_percentage, crop_percentage, corrupt_type):
    # assuming that the input image shape is [1, x], where x are the number of features.
    
    x = image.shape[1]
    k = 1 - crop_percentage
    kk = int(x*k)
    if corrupt_type == 'both':
        perturbed_image = Perturb(image, p = perturb_percentage)
        perturbed_image[:, kk:] = -1
        return perturbed_image
    
    if corrupt_type == 'perturb':
        perturbed_image = Perturb(image, p = perturb_percentage)
        return perturbed_image
    
    if corrupt_type == 'crop':
        k = 1 - crop_percentage
        kk = int(x*k)
        rin = random.randint(0, kk)
        rr = torch.rand(x)*255

        image[:, kk:] = rr[kk:]
        # image[:, rin:rin+kk] = -1
        return image