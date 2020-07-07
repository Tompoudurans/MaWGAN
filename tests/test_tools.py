#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:31:20 2020

@author: c1751832

"""
import numpy
import pandas
from src.tools.prepocessing import unnormalize
from src.tools.dataman import dagplot,simplesplit
from src.tools.fid import calculate_fid
from src.gans.tengan import dataGAN
from src.tools.sqlman import *
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

def test_dagplot():
    """
    test the ploting function
    """
    x=numpy.array([[1.0,1.2,1.3],[2.1,2.2,2.3]])
    y=testgan.create_fake(batch_size)
    dagplot(x, y)

def test_factorizing():
    """
    test the handiling cartogical data
    """
    data = pandas.DataFrame({'num':[1.0,1.2,1.3],'cat':['A','B','A']})
    newdata,indexs = factorizing(data)
    expected_index = numpy.array(['A', 'B'], dtype='object')
    expected_data = pandas.DataFrame({'num':[1.0,1.2,1.3],'cat':[1,2,1]})
    assert all(newdata == expected_data)
    assert all(indexs[0] == expected_index)


def test_sql_load_and_save():
    """
    test
    """
    df = pandas.DataFrame({1:[1.0,1.2,1.3],2:[2.1,2.2,2.3]})
    save_sql(df,"test")
    database, mean, std ,idexes = load_sql("test", "generated_data")
    dataset = unnormalize(database,mean,std)
    dataset = dataset.drop(columns=[0])
    assert all(dataset == df)

def test_simplesplit():
    data=numpy.array([[1.0,1.2],[1.3,1.4],[1.5,2.1],[2.2,2.3],[2.3,2.4]])
    split=simplesplit(data,40)
    assert len(split[0]) == 3
    assert len(split[1]) == 2
