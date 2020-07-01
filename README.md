# about

This code creates and train a GAN
This is a copy of david foster book 'deep generative models' which
has been modified to work with a numeric database such as the iris datasets taken from sklearn
soon this GAN would work on the DCWW dataset


# To install
Create a conda environment with all required libraries

```
$ conda env create -f environment.yml
$ source activate tfgan
```

# to test the code
```
$ pytest
```

# To run the code

```
$ python main.py
```

# Options

--mode (default=n):
(s)pyder: for developing the gan is outputed as a variable
(n)ormal: train gan and save on the storage
(m)arathon: train a gan for epochs > 50000
--filepath:
 enter the file name and location of the database and model
--epochs:
choose how long that you want to train
--dataset:
chose the dataset/table that the GAN will to train on this can't be a single letter
--model:
chose which model that you what to use
--opti:
chose the optimiser that you want to use
--noise:
chose the length of the noise vector
--batch:
chose how many fake data you want to make in one go
--layers:
chose the number of layers of each network
--clip:
if using wgan chose the cliping threshold
--core(default=0,type=int):
select number of core that you like to run
--sample (default=1,type=integer):
chose the number of generate data that you want: (samples*batch)
