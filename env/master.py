from makeholes import mkhole
from testenv import one_dataset
from grapher import set_graph
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

def set_exp(folder,datasets,muti):
    batch = [100,200,300]
    batch2 = [150,952,1250]
    for i in range(len(datasets)):
        one_dataset(datasets[i],muti,batch[i],batch2[i],folder)

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

def set_folder(folder,datanames):
    subprocess.run(["mkdir",folder])
    for dn in datanames:
        subprocess.run(["cp","base/00" + dn,folder])


if __name__ == '__main__':
    datasets = ["_percent_iris.csv","_Deprivation_percent.csv","_letter_percent.csv"]
    for i in range(90):
        set_folder("expset"+ str(i) + "/",datasets)
        set_mkhole("expset"+ str(i) + "/",datasets,True)
        set_exp("expset"+ str(i) + "/",datasets,False)
