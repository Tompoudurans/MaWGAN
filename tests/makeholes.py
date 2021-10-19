# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:08:23 2021

@author: drtro
"""

import pandas
import random
import numpy

file = "letter.csv"
percent = 0.50
data = pandas.read_csv(file)
x, y = data.shape
count = 0
aim = round(x * y * percent)
while count < aim:
    row = random.randint(0, x - 1)
    col = random.randint(0, y - 1)
    data.iloc[row, col] = numpy.nan
    count = data.isna().sum().sum()
    if count % 1000 == 0:
        print(count, "(", aim, ")/", x * y, sep="")
print(count, "(", aim, ")/", x * y, " final", sep="")
data.to_csv(str(int(percent * 100)) + "_letter_percent_.csv", index=False)
