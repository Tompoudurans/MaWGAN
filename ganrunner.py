from sklearn import datasets
from tengan import dataGAN
from dataman import dagpolt,show_loss_progress
from math import ceil
import numpy as np
set = input("set? 'w'/'i' ")
if set == 'i':
    database = datasets.load_iris()
elif set == 'w':
    database = datasets.load_wine()
else:
    quit()
batch = int(input('batch? '))
z = batch
no_field = len(database.data[1])
opti = input('opti? ')
mygan = dataGAN(opti,z,no_field,batch)
mygan.discriminator.summary()
mygan.model.summary()
filepath = input("load filepath: (or n?)")
if filepath != 'n':
    mygan.load_weights(filepath)
epochs = int(input('epochs? '))
if epochs > 0:
    step = int(ceil(epochs*0.001))
    mygan.train(database.data,batch,epochs,step)
    mygan.save_model()
    show_loss_progress(mygan.d_losses,mygan.g_losses)
noise = np.random.normal(0, 1, (batch, z))
generated_data = mygan.generator.predict(noise)
print(generated_data)
dagpolt(generated_data,database.data)
