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
clip = 0.1

testgan = dataGAN('adam', noise_vector, data, nodes, clip, layers)

def test_critic_layers():
    assert len(testgan.critic.layers) == layers

def test_gentrator_layers():
    assert len(testgan.generator.layers) == layers

def test_critic_input():
    assert testgan.critic.input_shape == (None, data)

def test_gentrator_input():
    assert testgan.generator.input_shape == (None, noise_vector)

def test_critic_output():
    assert testgan.critic.output_shape == (None, 1)

def test_gentrator_output():
    assert testgan.generator.output_shape == (None, data)

def test_critic_hidden():
    assert testgan.critic.layers[1].output_shape == (None, nodes)

def test_gentrator_hidden():
    assert testgan.generator.layers[1].output_shape == (None, nodes)

def test_model_input():
    assert testgan.model.input_shape == (None, noise_vector)

def test_model_output():
    assert testgan.model.output_shape == (None, 1)
