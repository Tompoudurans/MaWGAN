# -*- coding: utf-8 -*-

import ganrunner
import multiprocessing as mpg
import time
import subprocess
import pandas as pd

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
    parameters_list = [dataset, model, opti, cells, batch, layers, lambdas, rate]
    parameters, successfully_loaded = ganrunner.parameters_handeling(
        filename, parameters_list
    )
    epochs = int(epochs)
    beg = time.time()
    thegan, success = ganrunner.run(
        filename, epochs, parameters, successfully_loaded, database, batch#, True
    )
    totime = time.time() - beg
    print("time",totime)
    return thegan, details, totime

def fid_run(block):
    """
    run ingore-gan from specified dataset
    pramters as given in a list fromat
    this uses a custom made main fuction def above rather that the one in src
    """
    per,dataname,batch,batch2,folder = block
    cells = 150
    epochs = 10000
    print(per,"0% ------------------------------------------------------")
    thegan, details, totime = main(
        None,
        folder + str(per) + "0" + dataname,
        epochs,
        "linear",
        "adam",
        batch,#batch_size
        cells,
        5,#number_of_layers
        10,#lambdas
        0.0001,#learnig rate
    )
    full = ganrunner.tools.pd.read_csv(folder + "00" + dataname)
    pachal = full.sample(batch2)
    ls = []
    for i in range(100):
        syn = ganrunner.make_samples(
            thegan,
            None,
            batch2,
            None,#filepath
            details,
            None,#".csv",#extention
            False #show?
        )
        ls.append(ganrunner.tools.gpu_LS(pachal.to_numpy(),syn.to_numpy()))
    return ls, totime

def poolrun(dataname,batch,folder):
    block = []
    for i in range(10):
        block.append([i,dataname,batch,folder])
    p = mpg.Pool(processes=10)#--------------------------------------------------------------------------?
    result = p.map(fid_run,block)
    p.close()
    p.join()
    return result

def singal(dataname,batch,batch2,folder):
    i = 0
    fls, totime = fid_run([i,dataname,batch,batch2,folder])
    return fls, totime

def one_dataset(dataname,muti,batch,batch2,folder):
    if muti:
        fidata = poolrun(dataname,batch,folder)
    else:
        fidata, totime = singal(dataname,batch,batch2,folder)
    frame = ganrunner.tools.pd.DataFrame(fidata)
    frame.to_csv(folder + "ls_" + dataname,index=False)
    return totime

def set_exp(datasets,muti,batch,testname):
    times = []
    for i in range(len(batch)):
        folder = set_folder(testname + "/sample" + str(batch[i]),datasets)
        times.append(one_dataset(datasets,muti,batch[i],150,folder))
    frame = pd.DataFrame([batch, times])
    frame.to_csv(testname + "/times" + datasets,index=False)

def set_folder(folder,datanames):
    subprocess.run(["mkdir",folder])
    subprocess.run(["cp","00" + datanames,folder])
    return folder + "/"

if __name__ == '__main__':
    for testno in range(20):
        testname = "test" + str(testno)
        subprocess.run(["mkdir",testname])
        batches = [10,20,30,40,50,60,70,80,90]
        datanames = ["_percent_iris.csv","","_letter_percent.csv"]
        set_exp(datanames[0],False,batches,testname)
