import pandas
import matplotlib.pyplot as mp

def colect(arg,stuff,all):
    for i in arg:
        somthing = i + stuff
        peice = pandas.read_csv(somthing)
        peice = peice.drop(columns="Unnamed: 0")
        all = all.merge(peice)

def load_dataset(dataset):
    peice = pandas.read_csv(dataset)
    peice = peice.drop(columns="Unnamed: 0")
    return peice.transpose()


def make_box(data,place):
    mp.boxplot(data)
    mp.savefig(place + ".pdf")
    mp.clf()

def one_grath(filepath):
    to_plot = load_dataset(filepath)
    filename, extention = filepath.split(".")
    make_box(to_plot,filename)

def set_graph(folder,datasets):
    for file in datasets:
        one_grath(folder + "fids" + file)
