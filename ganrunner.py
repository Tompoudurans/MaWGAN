from sklearn import datasets
from tengan import dataGAN
from dataman import plotting
import numpy as np
batch = 100
iris = datasets.load_iris()
no_field = len(iris.data[1])
mygan = dataGAN('adam',batch,no_field,batch)
mygan.discriminator.summary()
mygan.model.summary()
mygan.train(iris.data,batch,1000)
noise = np.random.normal(0, 1, (100, 100))
generated_data = mygan.generator.predict(noise)
print(generated_data)
plotting(generated_data)
