import multiprocessing as mpg
import subprocess
import pandas
import ganrunner
import multiprocessing as mpg
import time
from makeholes import mkhole

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
        filename, epochs, parameters, successfully_loaded, database, batch, True
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
    folper,dataname,batch,batch2 = block
    cells = 150
    epochs = 15000
    thegan, details, totime = main(
        None,
        folper,
        epochs,
        "linear",
        "adam",
        batch,#batch_size
        cells,
        5,#number_of_layers
        10,#lambdas
        0.0001,#learnig rate
    )
    full = ganrunner.tools.pd.read_csv(dataname)
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
    return ls

def set_exp(folder,datasets,set):
    batch = 30
    batch2 = [150,952,1250]
    for i in range(len(datasets)):
        fidata = linexp(datasets[i],batch,batch2[i],set,folder)
        frame = ganrunner.tools.pd.DataFrame(fidata)
        frame.to_csv(folder + "ls_" + datasets[i],index=False)

def linexp(dataset,batch,batch2,set,folder):
    fls = []
    setname = "base/00" + dataset
    for per in range(7):
        dataname = folder + str(per) + "0" + dataset
        d = pandas.read_csv(dataname)
        d = d.dropna()
        no = d.count()[0]
        print("no:",no)
        if no <= 1:
            fls.append([0]*100)
        else:
            d.to_csv(dataname,index=False)
            if no < batch:
                fls.append(fid_run([dataname,setname,no,batch2]))
            else:
                fls.append(fid_run([dataname,setname,batch,batch2]))
    return fls

def poolhole(dataset,folder):
    block = []
    for i in dataset:
        block.append([i,folder])
    p = mpg.Pool(processes=6)
    p.map(mkhole,block)
    p.close()
    p.join()

def set_mkhole(folder,datanames,use_pools_for_making_holes):
    if use_pools_for_making_holes:
        poolhole(datanames,folder)
    else:
        nonmutihole(datanames,folder)

def set_folder(folder,datanames):
    subprocess.run(["mkdir",folder])
    for dn in datanames:
        subprocess.run(["cp","base/00" + dn,folder])

if __name__ == '__main__':
    datasets = ["_percent_iris.csv","_Deprivation_percent.csv","_letter_percent.csv"]
    mi=int(input("starting set: "))
    ma=int(input("ending set: "))
    for i in range(mi,ma):
        wkdir = "lineset" + str(i) + "/"
        set_folder(wkdir,datasets)
        set_mkhole(wkdir,datasets,True)
        set_exp(wkdir,datasets,str(i))
