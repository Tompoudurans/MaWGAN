#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:31:20 2020

@author: c1751832

"""
import numpy
from src.tools.prepocessing import unnormalize
from src.tools.fid import calculate_fid
from src.main.tengan import dataGAN
layers = 5
nodes = 20
data = 3
batch_size = 2
noise_vector = 10
testgan = dataGAN('adam', noise_vector, data, nodes, layers)

def test_fid():
    """
    testing fid distance it value should be greater than 0
    """
    testgan.load_weights('tests/testing')
    dataset=numpy.array([[1.0,1.2,1.3],[2.1,2.2,2.3]])
    data = testgan.create_fake(batch_size)
    f = calculate_fid(data,dataset)
    assert f > 0

def test_norm():
    """
    testing  unnormalize funtion it should give the orginal data
    """
    dataset=numpy.array([2.1,2.2,2.3])
    n = unnormalize(dataset,6,2)
    n = numpy.array(n)
    exp = numpy.array([10.2,10.4,10.6])
    for i in range(len(n)):
        assert n[i] == exp[i]
