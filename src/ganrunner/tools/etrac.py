# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
import scipy
import matplotlib.pyplot as mp
import numpy

def exctract(timedata, fac=20):
    """
    seprate noise from trend using fft
    """
    tran = timedata.transpose()
    for i in range(len(tran)):
        trend, compac = smoother(tran[i],fac)
        noise = tran[i] / trend
        if i == 0:
            trends = numpy.array([trend])
            noises = numpy.array([noise])
            compacs = numpy.array([compac])
        else:
            trends = numpy.append(trends,[trend],axis=0)
            noises = numpy.append(noises,[noise],axis=0)
            compacs = numpy.append(compacs,[compac],axis=0)
    return trends.transpose(), noises.transpose(), compacs.transpose()

def smoother(timedata,fac):
    """
    takes a fonier trasform of timedata and keeps fac trems before inverse trasform
    """
    lenght = len(timedata)
    wav = scipy.fft.rfft(timedata)
    compac = wav[0:fac]
    smooth = scipy.fft.irfft(compac,lenght)
    return smooth, compac

def expand(compac,lenght):
    tran = compac.transpose()
    smooth = []
    for i in tran:
        smooth.append(scipy.fft.irfft(i,lenght))
    return smooth

def movingav(self, timedata, alpha=0.3):
        timedata = pandas.DataFrame(timedata)
        trend = timedata.ewm(alpha=alpha, adjust=False).mean()
        noise = timedata / trend
        return trend, noise
