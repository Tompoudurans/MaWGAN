from sklearn import datasets
from tengan import dataGAN

iris = datasets.load_iris()

mygan = dataGAN('adam',iris.data,4)
