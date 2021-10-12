# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:08:23 2021

@author: drtro
"""

import pandas
import random
import numpy

def mkhole(block):
    file,folder = block
    for i in range(1,6):
        percent = i/10
        if i > 1:
            name = folder + str(i*10-10) + file
        else:
            name = folder + "00" + file
        print("opening:",name)
        data = pandas.read_csv(name)
        x, y = data.shape
        count = 0
        aim = round(x * y * percent)
        while count < aim:
            row = random.randint(0, x - 1)
            col = random.randint(0, y - 1)
            data.iloc[row, col] = numpy.nan
            count = data.isna().sum().sum()
            if count % 1000 == 0:
                print(count, "(", aim, ")/", x*y, sep="")
        print(count, "(", aim, ")/", x * y," final", sep="")
        data.to_csv(folder + str(int(percent * 100)) + file, index=False)
