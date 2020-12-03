#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:31:20 2020

@author: c1751832
"""
import ganrunner
import numpy
import os

layers = 5
nodes = 3
data = 3
batch_size = 3
noise_vector = 3
lambdas = 1
dataset = numpy.array([[1.0, 1.2, 1.3], [1.2, 1.1, 1.3], [1.4, 1.2, 1.5]])

testgan = ganrunner.wGANgp("adam", noise_vector, data, nodes, layers, lambdas, 0.00001)

def test_gan_training():
    """
    Tests the training algorithm of the GAN as the generator cannot be trained directly.
    This is done by taking an untrained sample, training the GAN 10 times, then taking a new sample
    and comparing the untrained sample and trained sample.
    The trained sample should provide a better result
    """
    numpy.random.seed(10)
    noise = numpy.random.normal(0, 1, (batch_size, noise_vector))
    untrained_fake = testgan.Generator.predict(noise)
    testgan.train(dataset, batch_size, 20)
    trained_fake = testgan.Generator.predict(noise)
    untrained = abs(untrained_fake - dataset)[0]
    trained = abs(trained_fake - dataset)[0]
    assert any(untrained > trained)


def test_save():
    """
    Tests the 'save' function
    """
    testgan.save_model("testing")
    assert os.stat("testing_generator.h5").st_size > 0
    assert os.stat("testing_critic.h5").st_size > 0
    assert os.stat("testing_model.h5").st_size > 0


def test_load():
    """
    Tests the 'load' function check the first weight
    """
    test = ganrunner.wGANgp("adam", noise_vector, data, nodes, layers, lambdas, 0.00001)
    generator_weight = test.Generator.get_weights()
    critic_weight = test.Critic.get_weights()
    test.load_weights("testing")
    assert (generator_weight[0] != test.Generator.get_weights()[0]).all()
    assert (critic_weight[0] != test.critic.get_weights()[0]).all()


def test_build():
    """
    This test checks that the GAN is well built and has the correct
    number of layers, input and output shape
    """
    assert len(testgan.Critic.layers) == layers
    assert len(testgan.Generator.layers) == layers
    assert testgan.Critic.input_shape == (None, data)
    assert testgan.Generator.input_shape == (None, noise_vector)
    assert testgan.critic.output_shape == (None, 1)
    assert testgan.Generator.output_shape == (None, data)
    assert testgan.critic.layers[1].output_shape == (None, nodes)
    assert testgan.Generator.layers[1].output_shape == (None, nodes)
