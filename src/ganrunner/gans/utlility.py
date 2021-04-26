import torch
import numpy

def vardag(x):
    #34.8 ms ± 1.53 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    li= []
    for i in x:
         torch.var(x) # calulate the variance of the varible i
         li.append(torch.var(i)) #append to a vector
    matrix = torch.tensor(li)
    return matrix.diag()# makes a diagonal matix with that vector

def vardrag(x):
    #4.54 ms ± 55.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    covmat = numpy.cov(x)# make covarince matrix
    varvec = covmat.diagonal()#put diagonal values in a vector
    vartensor = torch.tensor(varvec)
    vardiag = vartensor.diag()# makes a diagonal matix with that vector
    return vardiag.float()

def convert_item(item,sets):
    leng = sets[0].shape[0]
    return item.reshape(1,leng)

def weighted_entopy(x,y):
    pass
    #w = entopy(x)
    #mod = w*eucalian_distance(x-y)
    #return mod

def mahalanobis_distance(full,x,y):
    #D^2=(x-y)^T*V^-1*(x-y)
    V=vardrag(full)
    right=x-y # must be array or tensor
    mid=torch.inverse(V)
    left=right.T
    return torch.mm(left,torch.mm(mid,right))

def mutil_mahalanobis_distance(x,y):
    #D^2=(x-y)^T*V^-1*(x-y)
    V=vardrag(x.T)
    mid=torch.inverse(V)
    mutil_dis = []
    for i in range(len(x)):
        right=convert_item(x[i]-y[i],y)
        left=right.T
        mutil_dis.append(torch.mm(right,torch.mm(mid,left)))
    return torch.cat(mutil_dis)

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
