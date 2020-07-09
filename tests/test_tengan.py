#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:31:20 2020

@author: c1751832
"""
from src.gans.tengan import dataGAN
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
    testgan.save_model('test')

def test_load():
    """
    testing the load function
    """
    testgan.load_weights('test')

#this next few test checks the the gan is bulid well and has the correct
#number of layers, input and output shape

def test_discriminator_layers():
    assert len(testgan.discriminator.layers) == layers

def test_gentrator_layers():
    assert len(testgan.generator.layers) == layers

def test_discriminator_input():
    assert testgan.discriminator.input_shape == (None, data)

def test_gentrator_input():
    assert testgan.generator.input_shape == (None, noise_vector)

def test_discriminator_output():
    assert testgan.discriminator.output_shape == (None, 1)

def test_gentrator_output():
    assert testgan.generator.output_shape == (None, data)

def test_discriminator_hidden():
    assert testgan.discriminator.layers[1].output_shape == (None, nodes)

def test_gentrator_hidden():
    assert testgan.generator.layers[1].output_shape == (None, nodes)

def test_model_input():
    assert testgan.model.input_shape == (None, noise_vector)

def test_model_output():
    assert testgan.model.output_shape == (None, 1)
