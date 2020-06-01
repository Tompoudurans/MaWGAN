#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:31:20 2020

@author: c1751832
"""
from tengan import dataGAN
layers = 5
nodes = 20
data = 3
noise_vector = 10

testgan = dataGAN('adam', noise_vector, data, nodes, layers)

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
    