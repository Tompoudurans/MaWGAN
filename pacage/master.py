from makeholes import mkhole
from testenv import one_dataset
import os

def poolhole(dataset):
        block = []
        for i in range(1,6):
            block.append([i,dataset])
        p = mpg.Pool(6)
        result = p.map(mkhole,block)
        p.close()
        p.join()

def one_exp(datanames):
    batch = 100
    for dn in datanames:
        poolhole(dn)
        one_dataset(dn,muti,batch)
        batch = batch + 100

if __name__ == '__main__':
    one_exp(["_percent_iris.csv","_Deprivation_percent.csv","_letter_percent.csv"])
