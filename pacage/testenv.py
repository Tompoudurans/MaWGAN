# -*- coding: utf-8 -*-

import ganrunner
import multiprocessing as mpg
import time

def main(
    dataset,
    filepath,
    epochs,
    model,
    opti,
    noise,
    batch,
    layers,
    lambdas,
    rate
):
    """
    this is a custom made "main" funtion copied from src folder it does not have "core" or "samples"
    This code creates and trains a GAN.
    Core elements of this code are sourced directly from the David Foster book 'Deep generative models' and have been modified to work with a numeric database such as the iris datasets taken from the library 'sklearn'.
    """
    filename, extention = filepath.split(".")
    if extention == "csv":
        dataset = True
    ganrunner.tools.setup_log(filename + "_progress.log")
    parameters_list = [dataset, model, opti, noise, batch, layers, lambdas, rate]
    parameters, successfully_loaded = ganrunner.parameters_handeling(
        filename, parameters_list
    )
    epochs = int(epochs)
    database, details = ganrunner.load_data(parameters[0], filepath, extention)
    beg = time.time()
    thegan, success = ganrunner.run(
        filename, epochs, parameters, successfully_loaded, database
    )
    print("time",time.time() - beg)
    return thegan, details


def ls_run(block):
    """
    run ingore-gan from specified dataset
    pramters as given in a list fromat
    this uses a custom made main fuction def above rather that the one in src
    """
    per,dataname,batch,folder,epochs = block
    samples = [None,1,6,4]
    print(per,"0% ------------------------------------------------------")
    filename = folder + str(per) + "0" + dataname
    thegan, details= main(
        None,
        filename,
        epochs,
        "wgangp",
        "adam",
        batch,#noise_size
        batch,#batch_size
        5,#number_of_layers
        10,#lambdas
        0.0001,#learnig rate
    )
    ls_block = []
    for i in range(50):
        fake = ganrunner.make_samples(
            thegan,
            None,
            batch,
            samples[int(batch/100)],
            filename,
            details,
            "none",
            False,
        )
        full = ganrunner.tools.pd.read_csv(folder + "00" + dataname)
        if batch == 100:
            most = full.sample(n=100)
        else:
            most = full.sample(n=1200)
        og = most.to_numpy()
        sy = fake.to_numpy()
        ls = ganrunner.tools.gpu_LS(og,sy)
        ls_block.append(ls)
    return ls_block

def poolrun(dataname,batch,folder,epochs):
    block = []
    for i in range(6):
        block.append([i,dataname,batch,folder,epochs])
    p = mpg.Pool(processes=6)#--------------------------------------------------------------------------?
    result = p.map(ls_run,block)
    p.close()
    p.join()
    return result

def singal(dataname,batch,folder,epochs):
    set = []
    for i in range(10):
        set.append(ls_run([i,dataname,batch,folder,epochs]))
    return set

def one_dataset(dataname,muti,batch,folder,epochs):
    if muti:
        lsata = poolrun(dataname,batch,folder,epochs)
    else:
        lsata = singal(dataname,batch,folder,epochs)
    frame = ganrunner.tools.pd.DataFrame(lsata)
    frame.to_csv(folder + "ls" + dataname)
