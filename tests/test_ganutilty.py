from randomdatagen import make_data
from random import randint
import ganrunner

def test_hole():
    datahole, dataset = make_data(30,True)
    assert datahole.shape == (30,3)

def test_mask():
    datahole, dataset = make_data(30,True)
    template, masked_data = ganrunner.copy_format(datahole, dataset)
    #assert masked_data == datahole
    #test = masked_data == datahole
    #assert test.all()

def test_mahalanobis_distance():
    dis, x = make_data(30,False)
    dis, y = make_data(30,False)
    f = ganrunner.mahalanobis_distance(x,y)
    print(f)
