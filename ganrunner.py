from sklearn import datasets
from tengan import dataGAN
from dataman import dagpolt,show_loss_progress
from math import ceil
import numpy as np
from fid import calculate_fid
import flower

def marathon_mode(mygan,database,batch,noise_dim,filepath,epochs):
    """
    In marathon mode the gan is train for 50000 and subtracted from the number of epochs left
    then the gan model and the loss tracting is saved,
    the current loss traking is removed from ram and a new set of traing starts again
    at epoch 0 the result of the trainning is displayed and from there you can continue training if you wish.
    """
    while epochs > 0 :
        mygan.train(database,batch,50000,1000)
        mygan.save_model(filepath)
        np.savetxt(filepath + str(epochs) +'_d_losses.txt' ,mygan.g_losses)
        np.savetxt(filepath + str(epochs) +'_g_losses.txt' ,mygan.g_losses)
        mygan.d_losses,mygan.g_losses = [],[]
        epochs = epochs - 50000
        if epochs < 50000 and epochs > 0:
            print('almost done')
            mygan.train(database,batch,epochs,1000)
            break
        if epochs == 0:
            noise = np.random.normal(0, 1, (noise_dim,batch))
            generated_data = mygan.generator.predict(noise)
            print(generated_data)
            dagpolt(generated_data,database)
            calculate_fid(generated_data,database)
            epochs = int(input('contenue?, enter n* of epochs'))

def run(mode):
    """
    goes throught and store all the variables of dataGANclass
    then creates the gan load wieght if needed and train the GAN
    the option are fully expained on the README file
    """
    #select dataset
    set = input("set? 'w'/'i' ")
    if set == 'i':
        database = datasets.load_iris()
    elif set == 'w':
        database = datasets.load_wine()
    else:
        return None
    #estract data
    database = database.data
    batch = int(input('batch? '))
    #to not over complicate things, the noise_dim same as the batch size
    noise_dim = batch
    no_field = len(database[1])
    opti = input('opti? ')
    number_of_layers = int(input('layers? '))
    #create the GAN from the dataGAN
    mygan = dataGAN(opti,noise_dim,no_field,batch,number_of_layers)
    #print the stucture of the gan
    mygan.discriminator.summary()
    mygan.generator.summary()
    mygan.model.summary()
    #loads weights
    filepath = input("load filepath: (or n?) ")
    if filepath != 'n':
        try:
            mygan.load_weights(filepath)
        except OSError:# as 'Unable to open file':
            print('Error:404 file not found, starting from scrach')
    else:
        filepath = input('savepath? ')
    epochs = int(input('epochs? '))
    #marathon mode is not suitable when running less that 50000 epochs
    if epochs < 50000 and mode == 'm':
        print('epochs to small switch to normal ')
        mode = 'n'
    if epochs > 0:
        step = int(ceil(epochs*0.01))
        if mode == 'm':
            marathon_mode(mygan,database,batch,noise_dim,filepath,epochs)
        else:
            # train the gan acorrding to the number of epochs
            mygan.train(database,batch,epochs,step)
        if mode == 's':
            # in spyder mode the gan model return so it can be expermented on
            return mygan,database
        else:
            mygan.save_model(filepath)
        show_loss_progress(mygan.d_losses,mygan.g_losses)
    samples = input('samples? ')
    for s in range(int(samples)):
        noise = np.random.normal(0, 1, (noise_dim,batch))
        generated_data = mygan.generator.predict(noise)
        print(generated_data)
        dagpolt(generated_data,database)
        calculate_fid(generated_data,database)
    if mode == 's':
        return mygan

mode = input('mode?(s)pyder/(n)ormal/(m)arathon) ')
if mode == 's':
    gan,database = run(mode)
else:
    run(mode)
