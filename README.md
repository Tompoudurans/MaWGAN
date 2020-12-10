# About

This code creates and trains a GAN. Core elements of this code are sourced
directly from Green9's version of code from Improved Training of Wasserstein GANs" (https://arxiv.org/abs/1704.00028) and have been
modified to work with a numeric database such as the iris datasets taken
from the library 'sklearn'. Soon this GAN should work on the DCWW dataset.

## Installation

### Basic software requirements

To install, the following is assumed to be available:

- Python 3.7 (the software might work on other versions but it is only tested on 3.7 or 3.8).
- A recent version of git.

If those are not available please get in touch with the maintainer for alternative ways of installing

### To install

The following commands will download and install the software

    $ git clone https://github.com/Tompoudurans/dcwwgan
    $ cd dcwwgan
    $ python setup.py develop


To confirm installation was successful run the following:

    $ python -m ganrunner --help

### Test

The tool is tested using the Python testing framework pyrest. To install pytest:

    $ python -m pip install pytest

To run the tests:    

    $  python -m pytest


## To run the code

$ python -m ganrunner <Options>

filepath and epochs are maninitory, put the epochs to 0 to just read the the model

### Options

  --filepath   enter the file name and location of the database and model

  --epochs     choose how long that you want to train

  --dataset    choose the dataset/table that the GAN will train on

  --opti       choose the optimiser you want to use

  --batch      choose how many fake data you want to make in one go

  --layers     choose the number of layers of each network

  --clip       if using WGAN choose the clipping threshold

  --core      select the number of cores that you would like to run

  --sample    choose the number of generated data you want:(samples*batch)

  --rate     choose the learning rate of the model
