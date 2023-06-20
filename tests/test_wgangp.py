#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:31:20 2020

@author: c1751832
"""
import Mawgan
import numpy
import os
import torch

layers = 5
nodes = 3
data = 3
batch_size = 3
nodes_vector = 3
lambdas = 1
dataset = numpy.array([[1.0, 1.2, 1.3], [1.2, 1.1, 1.3], [1.4, 1.2, 1.5]])
testgan = Mawgan.MaWGAN(
    "adam", nodes_vector, nodes, layers, lambdas, 0.00001
)

def test_gan_training():
    """
    Tests the training algorithm of the GAN as the generator cannot be trained directly.
    This is done by taking an untrained sample, training the GAN 10 times, then taking a new sample
    and comparing the untrained sample and trained sample.
    The trained sample should provide a better result
    """
    nodes = torch.randn(batch_size, nodes_vector)
    untrained_fake = testgan.Generator(nodes).detach().numpy()
    testgan.train(dataset, 2, 20)
    trained_fake = testgan.Generator(nodes).detach().numpy()
    untrained = abs(untrained_fake - dataset)[0]
    trained = abs(trained_fake - dataset)[0]
    assert any(untrained != trained)

def test_save_load():
    """
    Tests the 'save' and 'load' function
    """
    testgan.save_model("testing")
    generator_weight_save = testgan.Generator.state_dict()
    critic_weight_save = testgan.Critic.state_dict()
    assert os.stat("testing_generator.pkl").st_size > 0
    assert os.stat("testing_critic.pkl").st_size > 0
    test = Mawgan.MaWGAN(
        "adam", nodes_vector, nodes, layers, lambdas, 0.00001
    )
    test.load_model("testing")
    assert all(generator_weight_save) == all(test.Generator.state_dict())
    assert all(critic_weight_save) == all(test.Critic.state_dict())


def test_build():
    """
    This test checks that the GAN is well built and has the correct
    number of layers, input and output shape
    """
    val = (layers * 2) - 1
    assert len(testgan.Critic) == val
    assert len(testgan.Generator) == val
    assert testgan.Critic[0].in_features == data
    assert testgan.Generator[0].in_features == nodes_vector
    assert testgan.Critic[2].in_features == nodes == testgan.Critic[0].out_features
    assert (
        testgan.Generator[2].in_features == nodes == testgan.Generator[0].out_features
    )
    assert testgan.Critic[(val - 1)].out_features == 1
    assert testgan.Generator[(val - 1)].out_features == data
