#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:31:20 2020

@author: c1751832

"""

from data.prepocessing import unnormalize
from fid import calculate_fid

def test_fid():
    """
    testing fid distance it value should be greater than 0
    """
    data = testgan.create_fake(batch_size)
    f = calculate_fid(data,dataset)
    assert f > 0

def broken_test_norm():
    """
    #testing  unnormalize funtion it should give the orginal data
    #"""
    n = unnormalize(dataset,6,2)
    exp = numpy.array([[8.2, 8.4, 8.6],[10.2,10.4,10.6]])
    assert all(n == exp)
