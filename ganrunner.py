from sklearn import datasets
from tengan import dataGAN

iris = datasets.load_iris()

mygan = dataGAN
mygan.build_adversarial()
mygan.train()
