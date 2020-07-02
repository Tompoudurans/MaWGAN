# about

This code creates and trains a GAN.
Core elements of this code are sourced directly from the David Foster book 'Deep generative models' and have been modified to work with a numeric database such as the iris datasets taken from the library 'sklearn'.
Soon this GAN should work on the DCWW dataset.


# To install
Create a conda environment with all required libraries:

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
(n)ormal: train the GAN and save on the storage
(m)arathon: train a GAN for > 50000 epochs
--filepath:
 enter the file name and location of the database and model
--epochs:
choose how long you want to train
--dataset:
choose the dataset/table that the GAN will train on, this can't be a single letter
--model:
choose which model you what to use
--opti:
choose the optimiser you want to use
--noise:
choose the length of the noise vector
--batch:
chose how many fake data you want to make in one go
--layers:
choose the number of layers of each network
--clip:
if using WGAN choose the clipping threshold
--core(default=0,type=int):
select the number of cores that you would like to run
--sample (default=1,type=integer):
choose the number of generated data that you want: (samples*batch)
