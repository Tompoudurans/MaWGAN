from sklearn import datasets
from tengan import dataGAN
from dataman import dagpolt,show_loss_progress
from math import ceil
import numpy as np
import flower

def run(mode):
    set = input("set? 'w'/'i' ")
    if set == 'i':
        database = datasets.load_iris()
    elif set == 'w':
        database = datasets.load_wine()
    else:
        return
    wt=input('with target? (t)')
    if wt == 't':
        datab = []
        for i in range(len(database.target)):
            datab.append(np.append(database.data[i],database.target[i]))
        datab = np.array(datab)
    else:
        datab = database.data
    batch = int(input('batch? '))
    z = batch
    no_field = len(datab[1])
    opti = input('opti? ')
    mygan = dataGAN(opti,z,no_field,batch)
    mygan.discriminator.summary()
    mygan.model.summary()
    filepath = input("load filepath: (or n?)")
    if filepath != 'n':
        mygan.load_weights(filepath)
    else:
        filepath = input('savepath? ')
    epochs = int(input('epochs? '))
    if epochs < 50000 and mode == 'm':
        print('epochs to small switch to normal')
        mode = 'n'
    if epochs > 0:
        step = int(ceil(epochs*0.01))
        try:
            if mode == 'm':
                while epochs > 0 :
                    mygan.train(datab,batch,50000,1000)
                    mygan.save_model(filepath)
                    np.savetxt(filepath + str(epochs) +'_d_losses.txt' ,mygan.g_losses)
                    np.savetxt(filepath + str(epochs) +'_g_losses.txt' ,mygan.g_losses)
                    mygan.d_losses,mygan.g_losses = [],[]
                    epochs = epochs - 50000
                    if epochs < 50000 and epochs > 0:
                        print('almost done')
                        mygan.train(datab,batch,epochs,1000)
                        break
                    if epochs == 0:
                        noise = np.random.normal(0, 1, (batch, z))
                        generated_data = mygan.generator.predict(noise)
                        print(generated_data)
                        dagpolt(generated_data,datab)
                        epochs = int(input('contenue?, enter n* of epochs'))
            else:
                mygan.train(datab,batch,epochs,step)
        except:
            if mode == 's':
                return mygan
            else:
                print('error try to save..')
                mygan.save_model(filepath)
                return
        if mode == 's':
            return mygan
        else:
            mygan.save_model(filepath)
        show_loss_progress(mygan.d_losses,mygan.g_losses)
    samples = input('samples? ')
    for s in range(int(samples)):
        noise = np.random.normal(0, 1, (batch, z))
        generated_data = mygan.generator.predict(noise)
        print(generated_data)
        dagpolt(generated_data,datab)
    if wt == 't':
        test_for_con = input('test y/n')
        if test_for_con == 'y':
            image,actual = [],[]
            for ge in range(len(generated_data)):
                im,ac=np.split(generated_data[ge],[4])
                image.append(im)
                actual.append(int(round(ac[0])))
            sets = flower.xref(database.data,database.target)
            print('seperating done')
            mods = flower.make_model()
            print('model done')
            truemod = flower.complexfit(mods,sets)
            print('fiting done')
            truemod.summary()
            flower.testmod(np.array(image),actual,truemod)
    if mode == 's':
        return mygan

mode = input('mode?(s)pyder/(n)ormal/(m)arathon)')
if mode == 's':
    gan = run(mode)
else:
    run(mode)
