from testenv import one_dataset
import multiprocessing as mpg
import subprocess

def poolhole(dataset,folder):
    block = []
    for i in dataset:
        block.append([i,folder])
    p = mpg.Pool(processes=6)
    p.map(mkhole,block)
    p.close()
    p.join()

def nonmutihole(datanames,folder):
    for dataset in datanames:
        block = [dataset,folder]
        mkhole(block)

def set_mkhole(folder,datanames,use_pools_for_making_holes):
    if use_pools_for_making_holes:
        poolhole(datanames,folder)
    else:
        nonmutihole(datanames,folder)

def set_exp(folder,datasets,muti,epochs):
    batch = 100
    for i in datasets:
        one_dataset(i,muti,batch,folder,epochs)
        batch = batch + 100

def chose(dataset):
    folder = input("folder? ")
    mutiop = input("use muti? 0/1")
    op = input("gan/holes/grath")
    if mutiop == "1":
        muti = True
    else:
        muti = False
    if op == "gan":
        set_exp(folder,datasets,muti)
    elif op == "hole":
        set_mkhole(folder,datasets,muti)
    elif op == "grath":
        set_graph(folder,datasets)
    else:
        pass

if __name__ == '__main__':
    folder = "set2to90/"
    muti = True
    datasets = ["_percent_iris.csv","_Deprivation_percent.csv","_letter_percent.csv"]
    set_exp(folder,datasets,False,15000)
