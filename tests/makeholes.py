# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:08:23 2021

@author: drtro
"""

import pandas
import random
import numpy

def mkhol(percent,data):
    data = pandas.DataFrame(data)
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
    return data

def main():
    percent = 0.5
    read = pandas.read_csv("letter.csv")
    da = mkhol(percent,read)
    da.to_csv(str(int(percent * 100)) + "_letter_percent_.csv", index=False)

if __name__ == '__main__':
    main()
