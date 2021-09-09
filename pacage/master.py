from makeholes import mkhole
#from testenv import one_dataset
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

def one_mkhole(folder,datanames,use_pools_for_making_holes):
    if use_pools_for_making_holes:
        poolhole(datanames,folder)
    else:
        nonmutihole(datanames,folder)

def one_exp(folder,datasets,muti):
    for i in datasets:
            one_dataset(i,muti,i*100,folder)

if __name__ == '__main__':
    folder = input("folder? ")
    mutiop = input("use muti? 0/1")
    holegan = input("gan/holes?")
    if mutiop == "1":
        muti = True
    else:
        muti = False
    datasets = ["_percent_iris.csv","_Deprivation_percent.csv","_letter_percent.csv"]
    if holegan == "gan":
        one_exp(folder,datasets,muti)
    elif holegan == "hole":
        one_mkhole(folder,datasets,muti)
    else:
        pass
