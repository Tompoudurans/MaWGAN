# about

This code creates and train a GAN
This is a copy of david foster book 'deep generative models' which
has been modified to work with a numeric database such as the iris datasets taken from sklearn
soon this GAN would work on the DCWW dataset

#to test the code
```
$ pytest
```

#To install
Create a conda environment with all required libraries
```
$ conda env create -f environment.yml
$ source activate tfgan
```
# To run the code
```
$ python ganrunner.py
```
# Menu explained

1)spyder: for developing the gan is outputed as a variable
normal: train gan and save on the storage
marathon: train a gan for epochs > 50000

2) use the wine database or the iris database?
3) batch? : chose how many fake data you want to make in one go
4) opti? : chose the optimiser that you want to use
5) layer? : chose the number of layers of each network
6) load? : chose a gan to load, press n to start from scratch
7) if not chose the name to save the network
8) choose how long that you want to train
then it will train
9) chose the number of generate data that you want: (samples*batch)
