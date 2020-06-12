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
batch_size = 1
noise_vector = 10
dataset=numpy.array([[1.0,1.2,1.3]])
#data.shape = (batch_size,data)

testgan = dataGAN('adam', noise_vector, data, nodes, layers)

def test_discriminator_training():
    untrained=testgan.discriminator.predict(dataset)
    testgan.train_discriminator(dataset, batch_size)
    trained=testgan.discriminator.predict(dataset)
    assert untrained < trained

def test_gan_training():
    noise = numpy.random.normal(0, 1, (batch_size, noise_vector))
    untrained_fake=testgan.generator.predict(noise)
    for i in range(10):
        testgan.train_discriminator(dataset, batch_size)
        testgan.train_generator(batch_size)
    trained_fake=testgan.generator.predict(noise)
    untrained = abs(untrained_fake - dataset)[0]
    trained = abs(trained_fake - dataset)[0]
    assert any(untrained > trained)

def test_fid():
    pass
