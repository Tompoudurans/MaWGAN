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
    batch,
    cells,
    layers,
    lambdas,
    rate,
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
    database, details = ganrunner.load_data(dataset, filepath, extention)
    parameters_list = [dataset, model, opti, batch, cells, layers, lambdas, rate]
    parameters, successfully_loaded = ganrunner.parameters_handeling(
        filename, parameters_list
    )
    epochs = int(epochs)
    beg = time.time()
    thegan, success = ganrunner.run(
        filename, epochs, parameters, successfully_loaded, database, batch
    )
    print("time",time.time() - beg)
    fake = None
    return thegan, fake, details

def fid_run(block):
    """
    run ingore-gan from specified dataset
    pramters as given in a list fromat
    this uses a custom made main fuction def above rather that the one in src
    """
    per,dataname,cells,folder = block
    batch = 150
    epochs = 5000
    print(per,"0% ------------------------------------------------------")
    a, b, c= main(
        None,
        folder + str(per) + "0" + dataname,
        epochs,
        "linear",
        "adam",
        cells,
        batch,#batch_size
        5,#number_of_layers
        10,#lambdas
        0.0001,#learnig rate
    )
    full = ganrunner.tools.pd.read_csv(folder + "00" + dataname)
    x = ganrunner.tools.get_norm(full)
    original = ganrunner.tools.pd.DataFrame(x[0])
    return a,c

def poolrun(dataname,batch,folder):
    block = []
    for i in range(10):
        block.append([i,dataname,batch,folder])
    p = mpg.Pool(processes=10)#--------------------------------------------------------------------------?
    result = p.map(fid_run,block)
    p.close()
    p.join()
    return result

def singal(dataname,batch,folder):
    set = []
    for i in range(10):
        if i == 0 and dataname == "_percent_iris.csv":
            pass
        else:
            fid_run([i,dataname,batch,folder])

def one_dataset(dataname,muti,batch,folder):
    if muti:
        fidata = poolrun(dataname,batch,folder)
    else:
        fidata = singal(dataname,batch,folder)
    frame = ganrunner.tools.pd.DataFrame(fidata)
    frame.to_csv(folder + "fids" + dataname)

if __name__ == '__main__':
    gan, det = fid_run([2,"_percent_iris.csv",100,""])