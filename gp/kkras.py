from sklearn import datasets
#from tengan import dataGAN
from tendupden import WGANGP
#from dataman import show_loss_progress
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as mp

def dagplot(x,y):
    fake = pd.DataFrame(x)
    real = pd.DataFrame(y)
    fake['dataset'] = ['fake']*len(x)
    real['dataset'] = ['real']*len(y)
    result = pd.concat([real, fake])
    sns.pairplot(result,hue='dataset')
    mp.show()

batch = 150
iris = datasets.load_iris()
no_field = len(iris.data[1])
#mygan = dataGAN('adam',batch,no_field,batch)
mygan = WGANGP(input_dim = no_field
        , critic_learning_rate =0.4
        , generator_initial_dense_layer_size = 150
        , generator_learning_rate = 0.4
        , optimiser = 'adam'
        , grad_weight = 1
        , z_dim = 0
        , batch_size = batch
        )
#opti, noise_dim, no_field, batch, number_of_layers,pam6

mygan.critic.summary()
#mygan.discriminator.summary()
mygan.model.summary()
mygan.train(iris.data,batch,500,'wgangp/',10,5)
noise = np.random.normal(0, 1, (150, 150))
generated_data = mygan.generator.predict(noise)
print(generated_data)
dagplot(generated_data,iris.data)
#show_loss_progress(mygan.d_losses,mygan.g_losses)
