#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:31:20 2020

@author: c1751832

"""
import numpy
from src.gans.tengan import dataGAN
from src.tools.prepocessing import unnormalize
from src.tools.fid import calculate_fid
from src.tools.dataman import dagplot


def test_fid():
    """
    Tests the fid distance, the fid value should be greater than 0.
    """
    dataset=numpy.array([[1.0,1.2,1.3],[2.1,2.2,2.3]])
    data=numpy.array([[2.1,2.2,2.3],[1.1,1.2,1.3]])
    f = calculate_fid(data,dataset)
    assert f > 0


def test_norm():
    """
    Tests the 'unnormalize' function, this function should output the orginal data
    """
    dataset=numpy.array([2.1,2.2,2.3])
    n = unnormalize(dataset,6,2)
    n = numpy.array(n)
    exp = numpy.array([10.2,10.4,10.6])
    for i in range(len(n)):
        assert n[i] == exp[i]

def test_dagplot():
    """
    Tests the plotting function by just running the code - still needs a assert line
    """
    x=numpy.array([[1.0,1.2,1.3],[2.1,2.2,2.3]])
    y=numpy.array([[2.1,2.2,2.3],[1.1,1.2,1.3]])
    dagplot(x, y,'test')
