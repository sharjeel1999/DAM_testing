import torch

import numpy as np
import copy

def hamming_score(vector1, vector2):
    assert len(vector1) == len(vector2), "Vectors must be of the same length"
    vector1 = torch.tensor(vector1, dtype=torch.float32)
    vector2 = torch.tensor(vector2, dtype=torch.float32)
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

    if corrupt_type == 'both':
        perturbed_image = Perturb(image, p = perturb_percentage)
        perturbed_image[:, k:] = -1
        return perturbed_image
    
    if corrupt_type == 'perturb':
        raise NotImplementedError("Perturb only is not implemented.")
    
    if corrupt_type == 'crop':
        raise NotImplementedError("Crop only is not implemented.")