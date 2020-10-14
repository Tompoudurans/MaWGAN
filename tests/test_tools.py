#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:31:20 2020

@author: c1751832

"""
import numpy
import pandas
import ganrunner
import os

layers = 5
nodes = 20
data = 3
batch_size = 2
noise_vector = 10
testgan = ganrunner.dataGAN("adam", noise_vector, data, nodes, layers)


def test_fid():
    """
    Tests the fid distance, the fid value should be greater than 0.
    """
    testgan.load_weights("tests/test")
    dataset = numpy.array([[1.0, 1.2, 1.3], [2.1, 2.2, 2.3]])
    data = testgan.create_fake(batch_size)
    f = ganrunner.calculate_fid(data, dataset)
    assert f > 0


def test_norm():
    """
    Tests the 'unnormalize' function, this function should output the orginal data
    """
    dataset = numpy.array([2.1, 2.2, 2.3])
    n = ganrunner.unnormalize(dataset, 6, 2)
    n = numpy.array(n)
    exp = numpy.array([10.2, 10.4, 10.6])
    for i in range(len(n)):
        assert n[i] == exp[i]


def test_dagplot():
    """
    Tests the plotting function by just running the code - still needs a assert line
    """
    x = numpy.array([[1.0, 1.2, 1.3], [2.1, 2.2, 2.3]])
    y = testgan.create_fake(batch_size)
    ganrunner.dagplot(x, y, "test")
    assert os.path.isfile("test_compare.pdf")


def test_encoding():
    data = pandas.read_csv("penguins_size.csv")
    new, bit = ganrunner.tools.encoding(data)
    assert new.dtypes.all() == "uint8"
    stuff = ganrunner.tools.decoding(new, bit)
    assert len(stuff) == len(data)


def test_sql_load_and_save():
    """
    Tests the import and export from SQL to python
    """
    df = pandas.DataFrame({1: [1.0, 1.2, 1.3], 2: [2.1, 2.2, 2.3]})
    ganrunner.save_sql(df, "tests/test", "replace")
    database, mean, std, idexes, col = ganrunner.load_sql(
        "tests/test", "generated_data"
    )
    dataset = ganrunner.unnormalize(database, mean, std)
    dataset = dataset.drop(columns=[0])
    assert all(dataset == df)


def test_simplesplit():
    """
    Tests splits the database into training data and testing data
    """
    data = numpy.array([[1.0, 1.2], [1.3, 1.4], [1.5, 2.1], [2.2, 2.3], [2.3, 2.4]])
    split = ganrunner.simplesplit(data, 40)
    assert len(split[0]) == 3
    assert len(split[1]) == 2
