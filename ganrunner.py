from sklearn import datasets
from tengan import dataGAN
from dataman import plotting,show_loss_progress
import numpy as np
batch = 100
iris = datasets.load_wine()
no_field = len(iris.data[1])
mygan = dataGAN('adam',batch,no_field,batch)
mygan.discriminator.summary()
mygan.model.summary()
mygan.train(iris.data,batch,100000)
noise = np.random.normal(0, 1, (100, 100))
generated_data = mygan.generator.predict(noise)
print(generated_data)
plotting(generated_data,iris.data)
show_loss_progress(mygan.d_losses,mygan.g_losses)
mygan.save_model()
