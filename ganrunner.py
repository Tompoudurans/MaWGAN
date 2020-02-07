from sklearn import datasets
from tengan import dataGAN
from dataman import dagpolt,show_loss_progress
import numpy as np
batch = 100
iris = datasets.load_iris()
no_field = len(iris.data[1])
mygan = dataGAN('RMSprop',batch,no_field,batch)
mygan.discriminator.summary()
mygan.model.summary()
mygan.discriminator.load_weights('Wgan_discriminator.h5')
mygan.generator.load_weights('Wgan_generator.h5')
mygan.model.load_weights('Wgan_model.h5')
mygan.train(iris.data,batch,50000,500)
noise = np.random.normal(0, 1, (100, 100))
generated_data = mygan.generator.predict(noise)
print(generated_data)
dagpolt(generated_data,iris.data)
show_loss_progress(mygan.d_losses,mygan.g_losses)
mygan.save_model()
