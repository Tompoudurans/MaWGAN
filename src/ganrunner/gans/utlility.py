import torch
import numpy

def vardag(x):
    li= []
    for i in x:
         torch.var(x)
         li.append(torch.var(i))
    matrix = torch.tensor(li)
    return matrix.diag()

def vardrag(x):
    covmat = numpy.cov(x)
    varvec = covmat.diagonal()
    vartensor = torch.tensor(varvec)
    return vartensor.diag()

def weighted_entopy(x,y):
    pass
    #w = entopy(x)
    #mod = w*eucalian_distance(x-y)
    #return mod

def mahalanobis_distance(x,y):
    #D^2=(x-y)^T*V^-1*(x-y)
    V=vardag(x)
    right=x-y # must be array or tensor
    mid=torch.inverse(V)
    left=right.T
    mid_right = torch.mm(mid,right)
    return torch.mm(left,mid_right)

def make_mask(data):
    """
    make mask hidding missing data
    """
    binary_mask = data.isnan()
    inverse_mask = torch.tensor(binary_mask, dtype=int)
    mask = 1 - inverse_mask
    return mask, binary_mask


def copy_format(template, data):
    """
    create a mask from the template and apply it to the data
    """
    mask, binary_mask = make_mask(template)
    masked_data = data * mask
    template[binary_mask] = 0
    return template, masked_data
