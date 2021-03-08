# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:08:23 2021

@author: drtro
"""

import pandas
import random
import numpy

file = "fulliris.csv"
percent = 0.65
data = pandas.read_csv(file)
x, y = data.shape
for i in range(round(x * y * percent)):
    row = random.randint(0, x - 1)
    col = random.randint(0, y - 1)
    data.iloc[row, col] = numpy.nan
print(data.isna().sum().sum(), "(", x * y * percent, ")/", x * y, sep="")
data.to_csv(str(int(percent * 100)) + "_percent_iris.csv")
