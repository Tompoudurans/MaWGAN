from sklearn import datasets
from tengan import dataGAN

iris = datasets.load_iris()

mygan = dataGAN
mygan.train(mygan,x_train=iris.data,batch_size=150,epochs=600)
