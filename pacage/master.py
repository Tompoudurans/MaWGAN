from makeholes import mkhole
from testenv import one_dataset
import multiprocessing as mpg
import subprocess

def poolhole(dataset,folder):
    block = []
    for i in range(1,6):
        block.append([i,dataset,folder])
    p = mpg.Pool(processes=6)
    p.map(mkhole,block)
    p.close()
    p.join()

def nonmutihole(dataset,folder):
    for i in range(1,6):
        block = [i,dataset,folder]
        mkhole(block)

def one_exp(folder,datanames,muti,use_pools_for_making_holes):
    batch = 100
    for dn in datanames:
        if use_multiprocessing_for_making_holes:
            poolhole(dn,folder)
        else:
            nonmutihole(dn,folder)
        one_dataset(dn,muti,batch,folder)
        batch = batch + 100

def folderman(folder,datanames):
    subprocess.run(["mkdir",folder])
    for dn in datanames:
        subprocess.run(["cp","00" + dn,folder])

if __name__ == '__main__':
    folder = input("folder? ")
    gmutiop = input("use muti for gan? 0/1")
    hmutiop = input("use muti for holes? 0/1")
    if hmutiop == "1":
        hmuti = True
    else:
        hmuti = False
    if gmutiop == "1":
        gmuti = True
    else:
        gmuti = False
    datasets = ["_percent_iris.csv","_Deprivation_percent.csv","_letter_percent.csv"]
    folderman(folder,datasets)
    one_exp(folder,datasets,gmuti,hmuti)
