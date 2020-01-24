# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:42:05 2019

@author: user
"""
#from torchvision import transforms
from torch import nn
import torch as tc
from sklearn import datasets
import numpy as np
#import dataman
from torch.autograd.variable import Variable
import random as rd
import matplotlib.pyplot as mp
iris = datasets.load_iris()

lossfun = nn.MSELoss()

def adjust(y):
    z = []
    for i in range(len(y)):
        z.append(np.long(y[i]))
    z = tc.tensor(z)
    return z

def ones(size):
    data = tc.ones(size, 1)
    return data

def zeros(size):
    data = tc.zeros(size, 1)
    return data

def genet(neurons):
    model = nn.Sequential(nn.Linear(neurons,neurons),
                          nn.Sigmoid(),
                          nn.Linear(neurons,neurons),
                          #nn.Sigmoid(),
                          nn.Linear(neurons,neurons),
                          #nn.Sigmoid(),
                          nn.Linear(neurons,neurons))
                          #nn.Sigmoid())
    optimizer = tc.optim.Adam(model.parameters())
    return model,optimizer


def disnet(field):
    model = nn.Sequential(nn.Linear(field,150),
                        nn.ReLU(),
                        nn.Linear(150,150),
                        nn.ReLU(),
                        nn.Linear(150,1),nn.Sigmoid())
    optimizer = tc.optim.Adam(model.parameters())
    #optimizer = tc.optim.Adadelta(model.parameters())
    return model,optimizer

def train_discriminator(mod,optimizer,loss,real,fake,size):
    optimizer.zero_grad()
    rpred = mod(real)
    fpred = mod(fake)
    target = Variable(tc.cat((zeros(size),ones(size)),0))
    error = loss(tc.cat((rpred,fpred),0),target)
    error.backward(retain_graph=True)
    optimizer.step()
    return mod,optimizer,error

def noise(vec):
    noise = []
    for i in range(vec):
        noise.append(rd.uniform(0,1))
        #noise.append(rd.randrange(0,1))
    return Variable(tc.tensor(noise))


def train_generator(optimizer, fake_data,discriminator,real_data,size,g):
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = lossfun(prediction,zeros(size))
    error.backward(retain_graph=True)
    optimizer.step()
    return optimizer,error


def disdata(real,fake):
    r = np.zeros(len(real))
    f = np.ones(len(fake))
    targ = np.concatenate((r,f))
    print(real,'real',fake,'fake')
    dat = np.concatenate((real,fake))
    return [tc.tensor(dat),tc.tensor(targ)]

def gan(cycle):
    fields = 4
    data = Variable(tc.tensor(iris.data)).float()
    size = len(data)
    #transforms.Normalize(0,1)(data)
    g,optg = genet(fields*size)
    d,optd = disnet(fields)
    discyc = 100
    t=0
    errorrec = []
    errorgen = []
    avdat = []
    for f in range(fields):
        avdat.append([])
    print('start')
#    bad = 0
   # for t in range(cycle):
    while t < cycle:
        t = t + 1
        try:
            fake_data = g(noise(fields*size))
            fake_data = fake_data.reshape(size,fields)
            #fake_data = fake_data*15
            for tau in range(discyc):
                d,optd,de = train_discriminator(d,optd,lossfun,data,fake_data,size)
                errorrec.append(de.item())
                if (tau+1) % 50 == True:
                    print('time=',t,':',tau,'discriminator:',de.item())
            opt,ge = train_generator(optg,fake_data,d,data,size,g)
            errorgen.append(ge.item())
            #if (t+1) % 10 == True or (t+1) == cycle:
            print('time=',t,':',discyc,'generator:',ge.item())
            s = sum(fake_data)/size
            print('avarage',s)
            for f in range(fields):
                avdat[f].append(s[f])
        except:
            t = 999999999999999999
            print('exit')
        #confac = dectectcon(s,prevs,confac)
        #prevs = s
    #    if t == (cycle-1):
    print('discriminator')
    mp.plot(errorrec)
    mp.show()
    print('generator')
    mp.plot(errorgen)
    mp.show()
    print('petals')
    for f in range(fields):
        mp.plot(avdat[f])
        mp.show()
    ploting(fake_data)
    return [d,g,optd,optg],fake_data

def ptest(net,x):
    plain=[x]*600
    plain = tc.tensor(plain)
    out = net(plain).reshape(150,4)
    print(out*10)
    
def ploting(tenso):
    tenso = tenso.detach()
    tenso = tenso.transpose(1,0)
    print('petals')
    mp.scatter(tenso[0],tenso[1])
    mp.show()
    print('sepals')
    mp.scatter(tenso[2],tenso[3])
    mp.show()


def dectectcon(now,pre,con):
    now = now.detach()
    try:
        pre = pre.detach()
        dif = sum(now - pre)
    except AttributeError:
        dif = 1
    if dif < 0.1 and dif > -0.1:
        con = con + 1
    else:
        con = 0
    print(dif,con)
    return con
    