from flower import simplesplit
from numpy import array

def test_split():
    testx = array([1,1,1,1])
    testy = array([1,1,1,1])
    sepratesum = sum(testx) + sum(testy)
    split = flower.simplesplit(testx,testy)
    unspit = sum(split[0]) + sum(split[1]) + sum(split[2]) + sum(split[3])
    assert unspit == sepratesum
