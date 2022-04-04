# -*- coding: utf-8 -*-

import ganrunner
import multiprocessing as mpg
import time
import logging
from torch import tensor
import pandas

def main(
    dataset, filepath, epochs, model, opti, batch, cells, layers, lambdas, rate,
):
    """
    this is a custom made "main" funtion copied from src folder it does not have "core" or "samples"
    This code creates and trains a GAN.
    Core elements of this code are sourced directly from the David Foster book 'Deep generative models' and have been modified to work with a numeric database such as the iris datasets taken from the library 'sklearn'.
    """
    filename, extention = filepath.split(".")
    ganrunner.tools.setup_log(filename + "_progress.log")
    database, details = ganrunner.load_data(filepath, extention)
    parameters_list = [dataset, model, opti, batch, cells, layers, lambdas, rate]
    parameters, successfully_loaded = ganrunner.parameters_handeling(
        filename, parameters_list
    )
    epochs = int(epochs)
    beg = time.time()
    thegan, success = letgo(
        filename, epochs, parameters, successfully_loaded, database,False
    )
    totime = time.time() - beg
    return thegan, details, totime

def letgo(filepath, epochs, parameters, successfully_loaded, database,usegpu=False):
    """
    Creates and trains a GAN from the parameters provided.
    """
    # select datase
    metadata, network, optimiser, noise_size, batch, number_of_layers, lambdas, learning_rate = parameters
    logging.info(filepath)
    input_dim = 4
    mygan = ganrunner.gans.decompGAN(
        optimiser,
        input_dim,
        int(noise_size),
        int(number_of_layers),
        int(lambdas),
        float(learning_rate),
        "linear")
    #mygan = ganrunner.load_gan_weight(filepath, mygan)
    if epochs > 0:
        #step = int(math.ceil(epochs * 0.1))
        step = 100
        tals = tensor([[0]*50,[1]*50,[2]*50]).reshape(1,150)
        mygan.train(database, tals[0], int(batch), int(epochs), False, step, 10, usegpu)
        mygan.save_model(filepath)
    return mygan, True

def fid_run(block):
    """
    run ingore-gan from specified dataset
    pramters as given in a list fromat
    this uses a custom made main fuction def above rather that the one in src
    """
    per, dataname, batch, batch2, folder = block
    cells = 100
    epochs = 15000
    thegan, details, totime = main(
        None,
        folder + str(per) + "0" + dataname,
        epochs,
        "linear",
        "adam",
        batch,  # batch_size
        cells,
        5,  # number_of_layers
        10,  # lambdas
        0.0001,  # learnig rate
    )
    set0 = pandas.DataFrame(thegan.create_fake(50,0))
    set2 = pandas.DataFrame(thegan.create_fake(50,2))
    set0.to_csv(folder + "set0_" + dataname, index=False)
    set2.to_csv(folder + "set2_" + dataname, index=False)
    return None


def poolrun(dataname, batch, batch2, folder):
    block = []
    for i in range(10):
        block.append([i, dataname, batch, batch2, folder])
    p = mpg.Pool(
        processes=10
    )  # --------------------------------------------------------------------------?
    result = p.map(fid_run, block)
    p.close()
    p.join()
    return result


def singal(dataname, batch, batch2, folder):
    fls = []
    for i in [0]:  # range(1,10):
        fls.append(fid_run([i, dataname, batch, batch2, folder]))
    return fls


def one_dataset(dataname, muti, batch, batch2, folder):
    if muti:
        fidata = poolrun(dataname, batch, batch2, folder)
    else:
        fidata = singal(dataname, batch, batch2, folder)
