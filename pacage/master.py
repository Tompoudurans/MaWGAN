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

def one_exp(folder,datanames,use_pools_for_making_holes):
    if use_pools_for_making_holes:
        poolhole(datanames,folder)
    else:
        nonmutihole(datanames,folder)


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
        batch = 100
        one_dataset(dn,gmuti,batch,folder)
    elif holegan == "hole":
        one_exp(folder,datasets,muti)
    else:
        pass
