import pandas
import matplotlib.pyplot as mp

def colect(arg,stuff,all):
    for i in arg:
        somthing = i + stuff
        peice = pandas.read_csv(somthing)
        peice = peice.drop(columns="Unnamed: 0")
        all = all.merge(peice)

def make_box(data,place):
    tran = data.transpose()
    mp.boxplot(tran)
    mp.savefig(place + ".pdf")
