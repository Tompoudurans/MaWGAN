from sklearn import datasets
from tengan import dataGAN
from dataman import dagpolt,show_loss_progress
from fid import calculate_fid
import numpy as np
batch = 150
iris = datasets.load_iris()
no_field = len(iris.data[1])
mygan = dataGAN('RMSprop',batch,no_field,batch,0.2)
mygan.discriminator.summary()
mygan.model.summary()
try:
    mygan.discriminator.load_weights('Wgan_discriminator.h5')
    mygan.generator.load_weights('Wgan_generator.h5')
    mygan.model.load_weights('Wgan_model.h5')
except:
    print('no file found strating from scrach')
mygan.train(iris.data,batch,50,10)
mygan.save_model()
#print(generated_data)
for i in range(3):
    noise = np.random.normal(0, 1, (150, 150))
    generated_data = mygan.generator.predict(noise)
    calculate_fid(generated_data,iris.data)
    dagpolt(generated_data,iris.data)
#show_loss_progress(mygan.d_losses,mygan.g_losses)
