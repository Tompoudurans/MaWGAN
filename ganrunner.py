from sklearn import datasets
from tengan import dataGAN
from dataman import dagpolt,show_loss_progress
import numpy as np
batch = 150
z = batch
iris = datasets.load_iris()
no_field = len(iris.data[1])
mygan = dataGAN('adam',z,no_field,batch)
mygan.discriminator.summary()
mygan.model.summary()
mygan.train(iris.data,batch,100)
noise = np.random.normal(0, 1, (batch, z))
generated_data = mygan.generator.predict(noise)
print(generated_data)
dagpolt(generated_data,iris.data)
show_loss_progress(mygan.d_losses,mygan.g_losses)
mygan.save_model()
