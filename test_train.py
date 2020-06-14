#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:31:20 2020

@author: c1751832
"""
from tengan import dataGAN
from data.prepocessing import unnormalize
from fid import calculate_fid
import numpy
layers = 5
nodes = 20
data = 3
batch_size = 2
noise_vector = 10
dataset=numpy.array([[1.0,1.2,1.3],[2.1,2.2,2.3]])

numpy.random.seed(seed=10)

testgan = dataGAN('adam', noise_vector, data, nodes, layers)

def test_discriminator_training():
    """
    test the training algorithm of the discriminator
    this done by taking untrained sample,
    training the discriminator and take new sample
    then compare the untrained sample and trained sample
    the trained sample should provide be better result
    """
    untrained=testgan.discriminator.predict(dataset)
    testgan.train_discriminator(dataset, batch_size)
    trained=testgan.discriminator.predict(dataset)
    assert all(untrained < trained)

def test_gan_training():
    """
    test the training algorithm of the GAN as the generator can not be trained directly
    this done by taking untrained sample, training the GAN 10 times and take new sample
    then compare the untrained sample and trained sample
    the trained sample should provide be better result
    """
    noise = numpy.random.normal(0, 1, (batch_size, noise_vector))
    untrained_fake=testgan.generator.predict(noise)
    for i in range(10):
        testgan.train_discriminator(dataset, batch_size)
        testgan.train_generator(batch_size)
    trained_fake=testgan.generator.predict(noise)
    untrained = abs(untrained_fake - dataset)[0]
    trained = abs(trained_fake - dataset)[0]
    assert any(untrained > trained)

def test_save():
    """
    testing the save function
    """
    testgan.save_model('test_')

def test_load():
    """
    testing the load function
    """
    testgan.load_weights('test_')

def test_fid():
    """
    testing fid distance it value should be greater than 0
    """
    data = testgan.create_fake(batch_size)
    f = calculate_fid(data,dataset)
    assert f > 0

#def test_norm():
    #"""
    #testing  unnormalize funtion it should give the orginal data
    #"""
#    n = unnormalize(dataset,6,2)
#    exp = numpy.array([[8.2, 8.4, 8.6],[10.2,10.4,10.6]])
#    assert all(n == exp)
