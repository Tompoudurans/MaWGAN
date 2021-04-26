from randomdatagen import make_data
from random import randint
import ganrunner
import torch

def test_hole():
    datahole, dataset = make_data(30,True)
    assert datahole.shape == (30,3)

def test_mask():
    datahole, dataset = make_data(30,True)
    template, masked_data = ganrunner.copy_format(datahole, dataset)
    f = masked_data.round() == template.round()
    assert f.all()


def test_mahalanobis_distance():
    dis, full = make_data(10,False)
    dis, y = make_data(1,False)
    item = full[2]
    x = ganrunner.convert_item(item,full)
    mdis = ganrunner.mahalanobis_distance(full.T,x.T,y.T)
    assert mdis.shape == torch.Size([1, 1])
    
def test_multi_mhal_distance():
    dis, x = make_data(10,False)
    dis, y = make_data(10,False)
    mdis = ganrunner.mutil_mahalanobis_distance(x,y)
    return mdis