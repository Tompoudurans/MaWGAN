#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:31:20 2020

@author: c1751832
"""
from tengan import dataGAN
import numpy
layers = 5
nodes = 20
data = 3
batch_size = 2
noise_vector = 10
dataset=numpy.array([[1.0,1.2,1.3],[2.1,2.2,2.3]])

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

def test_build():
    """
    this test checks the the gan is bulid well and has the correct
    number of layers, input and output shape
    """
    assert len(testgan.discriminator.layers) == layers
    assert len(testgan.generator.layers) == layers
    assert testgan.discriminator.input_shape == (None, data)
    assert testgan.generator.input_shape == (None, noise_vector)
    assert testgan.discriminator.output_shape == (None, 1)
    assert testgan.generator.output_shape == (None, data)
    assert testgan.discriminator.layers[1].output_shape == (None, nodes)
    assert testgan.generator.layers[1].output_shape == (None, nodes)
    assert testgan.model.input_shape == (None, noise_vector)
    assert testgan.model.output_shape == (None, 1)
