from sklearn import datasets
from tengan import dataGAN
import numpy as np
batch = 100
iris = datasets.load_wine()
no_field = 13
mygan = dataGAN('adam',batch,no_field)
mygan.train(iris.data,batch,1000)
noise = np.random.normal(0, 1, (100, 100))
print(mygan.generator.predict(noise))
