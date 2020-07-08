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
    Tests the training algorithm of the discriminator.
    This is done by taking an untrained sample,
    training the discriminator, then taking a new sample
    and comparing the untrained sample and trained sample.
    The trained sample should provide a better result.
    """
    numpy.random.seed(11)
    untrained=testgan.discriminator.predict(dataset)
    testgan.train_discriminator(dataset, batch_size)
    trained=testgan.discriminator.predict(dataset)
    assert all(untrained < trained)

def test_gan_training():
    """
    Tests the training algorithm of the GAN as the generator cannot be trained directly.
    This is done by taking an untrained sample, training the GAN 10 times, then taking a new sample
    and comparing the untrained sample and trained sample.
    The trained sample should provide a better result
    """
    numpy.random.seed(10)
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
    Tests the 'save' function
    """
    testgan.save_model('test')

def test_load():
    """
    Tests the 'load' function
    """
    testgan.load_weights('test')

def test_build():
    """
    This test checks that the GAN is well built and has the correct
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
