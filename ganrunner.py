from sklearn import datasets
from tengan import dataGAN

iris = datasets.load_iris()

mygan = dataGAN('adam',100)
mygan.train(iris.data,1,100)
