
# Pyhton code of MaWGAN

## About

This is the code for "MaWGAN: a Generative Adversarial Network to create synthetic
data from datasets with missing data"
https://orca.cardiff.ac.uk/id/eprint/148018/1/electronics-11-00837.pdf
If you use this code, please cite:
```
@article{poudevigne2022mawgan,
  title={MaWGAN: a generative adversarial network to create synthetic data from datasets with missing data},
  author={Poudevigne-Durance, Thomas and Jones, Owen Dafydd and Qin, Yipeng},
  journal={Electronics},
  volume={11},
  number={6},
  pages={837},
  year={2022},
  publisher={MDPI}
}
```
Note: This code is used Green9's WGAN-gp code as a staring point which is now heavly modified 

## Installation

### Basic software requirements

To install, the following is assumed to be available:

- Python 3.8+
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

      --filepath TEXT   enter the file name and location of the database and model
      --epochs TEXT     choose how long that you want to train
      --dataset TEXT    choose the dataset/table that the GAN will train on
      --opti TEXT       choose the optimiser you want to use
      --node TEXT      choose the number nodes per layer
      --batch TEXT      choose how many fake data you want to make in one go
      --layers TEXT     choose the number of layers of each network
      --lambdas FLOAT   learning penalty
      --sample INTEGER  choose the number of generated data you want
      --rate FLOAT      choose the learing rate of the model
      --help            Show this message and exit.

examples:
.db file without data missing, and displaying a compare graph

    $ python -m ganrunner --model=wgangp --filepath=iris.db --opti=adam --nodes=100 --batch=100 --layers=5 --epochs=1000 --dataset=all --rate=0.0001 --lambdas=10 --sample=300

.csv file with data missing

    $ python -m ganrunner --model=wgangp --filepath=10_Deprivation_percent.csv --opti=adam --nodes=200 --batch=300 --layers=5 --epochs=1000 --rate=0.0001 --lambdas=10 --sample=200
