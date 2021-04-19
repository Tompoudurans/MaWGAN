import torch

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
